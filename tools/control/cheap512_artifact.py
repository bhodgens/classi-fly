"""Build + validate the CHEAP deployable artifact: 512-neuron synthetic .fly.

The size co-sweep (tools/control/size_cosweep_results.json) found that a
512-neuron synthetic reservoir (lognormal weights, fan-in 6, generator seed 7,
spectral radius rescaled to exactly 0.5) matches the real 2,952-neuron larval
connectome on the lane-1 control task: 100% / 18.2 steps (blink3) and
100% / 37.6 (blink8) vs the connectome's 100% / 17.8 (blink3). This script
turns that config into a REAL .fly artifact through the Go toolchain and
validates it end to end.

Phases
------
1. build   : re-train the readout with the frozen co-sweep protocol (identical
             construction: size_cosweep.build_W + reservoir_imitation
             machinery; 4000 train + 1000 val pairs, seeds 5000+/6000+; ridge
             lambda swept and picked on held-out validation; final refit on
             train+val). One readout per blink mode, matching how the sweep
             scored the config. A float-reference sanity eval (200 episodes)
             must reproduce the sweep row before packing.
2. export  : write Contract-3 JSON exports:
             - adjacency: the synthetic CSR (float weights; the packer
               quantizes int8 at max|w|/127, dequant error <= scale/2).
             - readout: classes ['turn','thrust'], W = ridge solution^T,
               thresholds 0 (never abstain). FORMAT REUSE NOTE: .fly readouts
               are K-class classification heads; the control task has 2
               continuous outputs. We store the regression solution in the
               classification shape and evaluate it as regression
               (action = tanh(logits)); the softmax/abstain layer is unused.
             - pack block: embed_dim = 7 (actual sensor width), steps = 4,
               input_mode matrix (the lane-1 int8 projection, in_scale 0.02 -
               byte-exact round trip).
3. pack    : `classi-fly pack` (Go) -> data/cheap512_<mode>.fly, then
             `info` + a `classify` spot-check against Python argmax.
4. eval    : 200-episode lane-1 protocol (seed0=1000) with the packed bytes
             loaded via tools/eval/flyio.py. Deviations from flyio's reference
             Classify, required by the control loop and documented here:
             (a) state is carried across world steps and reset per episode
             (flyio's Classify is stateless per call);
             (b) the recurrence honors the header decay
             (h = tanh(decay*h + A h + u); flyio's _propagate omits decay -
             the Go runtime reservoir/classify.go applies it);
             (c) the readout is consumed as regression (tanh(logits)), not
             softmax-argmax. Everything else (CSR, int8 dequant, input
             projection) is exactly the reference reader's.

Run:  python3 tools/control/cheap512_artifact.py
"""

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "control"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

import flyio  # noqa: E402
import reservoir_imitation as ri  # noqa: E402
import size_cosweep as cs  # noqa: E402
from world import run_episode  # noqa: E402

N, RHO, DIST = 512, 0.5, "lognormal"
TRAIN_PAIRS, VAL_PAIRS = 4000, 1000
TRAIN_SEED0, VAL_SEED0 = 5000, 6000
MAX_EPISODES_COLLECT = 2000
EVAL_EPISODES, EVAL_SEED0 = 200, 1000
DECAY, STEPS, EMBED_DIM = 0.8, 4, 7
PROJ_SCALE = 0.02
MODES = [("blink3", 3), ("blink8", 8)]
CLASSES = ["turn", "thrust"]

DATA = REPO / "data"
RESULTS_PATH = REPO / "tools" / "control" / "cheap512_results.json"
BIN = Path("/tmp/classi-fly-cheap512")


# --------------------------------------------------------------------------- #
# Contract-3 export writers
# --------------------------------------------------------------------------- #
def write_exports(W, proj, Wout, bout, tag):
    indptr, indices, weights = W.indptr, W.indices, W.data
    adj = {
        "name": f"synthetic-{N}-{DIST}-rho{RHO}",
        "neurons": N,
        "edges": int(len(indices)),
        "indptr": [int(v) for v in indptr],
        "indices": [int(v) for v in indices],
        "weights": [float(v) for v in weights],
        "source": "synthetic-reservoir",
        "license": "CC0-1.0",
        "attribution": (f"synthetic lognormal fan-in {cs.FAN_IN}, seed "
                        f"{cs.GEN_SEED}, spectral radius {RHO} (size co-sweep "
                        f"cheapest deployable)"),
    }
    ro_scale = float(np.max(np.abs(Wout)) / 127.0)
    ro = {
        "classes": CLASSES,
        "W": [[float(v) for v in row] for row in Wout.T],  # (neurons, K)
        "bias": [float(v) for v in bout],
        "threshold": [0.0, 0.0],           # regression reuse: never abstain
        "weight_scale": ro_scale,          # trainer convention: max|W|/127
        "readout_scale": ro_scale,
    }
    in_w = np.rint(proj[:EMBED_DIM] / PROJ_SCALE).astype(np.int8)  # byte-exact
    pack = {
        "embed_dim": EMBED_DIM,
        "steps": STEPS,
        "input_mode": "matrix",
        "in_w": [int(v) for v in in_w.ravel()],
        "in_scale": PROJ_SCALE,
    }
    adj_path = DATA / f"cheap512_adjacency.json"
    adj_path.write_text(json.dumps(adj))
    ro_path = DATA / f"cheap512_readout_{tag}.json"
    ro_path.write_text(json.dumps(ro))
    pb_path = DATA / f"cheap512_pack.json"
    pb_path.write_text(json.dumps(pack))
    return adj_path, ro_path, pb_path


# --------------------------------------------------------------------------- #
# stateful controller over the PACKED bytes (flyio reader + control semantics)
# --------------------------------------------------------------------------- #
class FlyController:
    def __init__(self, path):
        m = flyio.Reservoir.Load(path)._m
        assert m.header["decay"] and abs(m.header["decay"] - DECAY) < 1e-12
        vals = m.weights.astype(np.float64) * m.weight_scale
        self.W = sparse.csr_matrix((vals, m.indices, m.indptr),
                                   shape=(m.n, m.n))
        self.win = m.in_w.astype(np.float64) * m.in_scale      # (d, n)
        self.Wout = m.readout.astype(np.float64) * m.readout_scale  # (n, k)
        self.bias = m.bias
        self.decay, self.steps, self.n = m.header["decay"], m.steps, m.n
        self.reset()

    def reset(self):
        self.s = np.zeros(self.n)

    def __call__(self, obs):
        u = np.asarray(obs, dtype=np.float64) @ self.win
        s = self.s
        for _ in range(self.steps):
            s = np.tanh(self.decay * s + (self.W @ s) + u)
        self.s = s
        return np.tanh(self.Wout.T @ s + self.bias)  # regression reuse


def eval_fly(path, mode, bp, n_episodes, seed0):
    ctl = FlyController(path)
    succ, steps = 0, []
    for i in range(n_episodes):
        r = run_episode(lambda obs: ctl.__call__(obs), seed=seed0 + i,
                        mode=mode, blink_period=bp)
        if r["success"]:
            succ += 1
            steps.append(r["steps"])
        ctl.reset()
    return {"success_rate": succ / n_episodes,
            "mean_steps": round(float(np.mean(steps)), 1) if steps else None,
            "timeouts": n_episodes - succ, "episodes": n_episodes}


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    results = {"config": {"neurons": N, "rho": RHO, "dist": DIST,
                          "decay": DECAY, "steps": STEPS, "embed_dim": EMBED_DIM,
                          "eval": f"{EVAL_EPISODES} eps seed0={EVAL_SEED0}"},
               "modes": {}, "go": {}}

    # go build once
    subprocess.run(["go", "build", "-o", str(BIN), "./cmd/classi-fly"],
                   cwd=REPO, check=True)

    # one synthetic graph + projection, shared across modes (co-sweep exact)
    W = cs.build_W(N, RHO, DIST)
    proj = ri.make_projection(N)
    (DATA / "cheap512_adjacency.json").unlink(missing_ok=True)

    for mode, bp in MODES:
        print(f"=== {mode} ===", flush=True)
        (Xtr, _), Ytr = ri.collect_pairs(W, proj, mode, bp, TRAIN_SEED0,
                                         MAX_EPISODES_COLLECT, TRAIN_PAIRS,
                                         verbose=False)
        (Xva, _), Yva = ri.collect_pairs(W, proj, mode, bp, VAL_SEED0, 500,
                                         VAL_PAIRS, verbose=False)
        Wout, bout, info = ri.fit_representation(Xtr, Ytr, Xva, Yva,
                                                 verbose=False, label=mode)

        # float-reference sanity eval (pre-quantization, frozen lane-1 code)
        m_float = ri.run_many_factory(
            lambda: ri.ReservoirController(W, proj, Wout, bout),
            n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0, mode=mode,
            blink_period=bp)

        adj_p, ro_p, pb_p = write_exports(W, proj, Wout, bout, mode)
        fly_path = DATA / f"cheap512_{mode}.fly"
        subprocess.run([str(BIN), "pack", "--adjacency", str(adj_p),
                        "--readout", str(ro_p), "--pack", str(pb_p),
                        "--out", str(fly_path), "--decay", str(DECAY)],
                       cwd=REPO, check=True)

        # Go info + classify spot-check: same argmax as Python logits
        m_ctl = FlyController(fly_path)
        env = __import__("world").Arena(seed=4242, mode=mode, blink_period=bp)
        obs = env.reset()
        spot = []
        for _ in range(5):
            h = np.zeros(N)
            u = np.asarray(obs) @ (proj[:EMBED_DIM])
            for _ in range(STEPS):
                h = np.tanh(DECAY * h + (W @ h) + u)
            py_cls = CLASSES[int(np.argmax(h @ Wout.T + bout))]
            emb = ",".join(f"{v:.6f}" for v in obs)
            out = subprocess.run(
                [str(BIN), "classify", str(fly_path), "--embedding", emb],
                cwd=REPO, capture_output=True, text=True, check=True)
            go_cls = json.loads(out.stdout)["class"]
            spot.append({"python": py_cls, "go": go_cls, "agree": py_cls == go_cls})
            obs, _r, done, _info = env.step(np.array([0.5, 0.5]))  # 0.5,0.5 clipped in step()
            if done:
                break

        m_fly = eval_fly(fly_path, mode, bp, EVAL_EPISODES, EVAL_SEED0)
        results["modes"][mode] = {
            "lam": info["best_lambda"], "val_mse": info["val_mse"],
            "float_reference": {"success_rate": m_float["success_rate"],
                                "mean_steps": m_float["mean_steps_to_success"]},
            "fly_artifact": m_fly,
            "go_classify_spot_check": spot,
            "fly_bytes": fly_path.stat().st_size,
        }
        print(json.dumps(results["modes"][mode]["fly_artifact"]), flush=True)
        print("spot: " + json.dumps(spot), flush=True)

    # size comparison + reference numbers
    prod = DATA / "prod_model.fly"
    results["sizes"] = {"cheap512_blink3_fly":
                        (DATA / "cheap512_blink3.fly").stat().st_size,
                        "cheap512_blink8_fly":
                        (DATA / "cheap512_blink8.fly").stat().st_size,
                        "prod2952_fly": prod.stat().st_size}
    results["references"] = {
        "real_connectome_blink3": {"success_rate": 1.0, "mean_steps": 17.8},
        "cosweep_512_row": {"success_rate": 1.0, "mean_steps": 18.2,
                            "blink8": {"success_rate": 1.0, "mean_steps": 37.6}},
    }
    b3, b8 = results["modes"]["blink3"]["fly_artifact"], \
        results["modes"]["blink8"]["fly_artifact"]
    ok = (b3["success_rate"] >= 0.95 and b3["mean_steps"] <= 25.0
          and b8["success_rate"] >= 0.95)
    results["verdict"] = ("CHEAP ARTIFACT MATCHES" if ok
                          else "CHEAP ARTIFACT DEGRADES")
    results["runtime_seconds"] = round(time.time() - t0, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nVERDICT: {results['verdict']}  sizes={results['sizes']}")
    print(f"wrote {RESULTS_PATH} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

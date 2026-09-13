"""Lane 1 harness: a 2D phototaxis arena for testing a fixed-graph controller.

Motivation (docs/DIRECTIONS.md lane 1): the fly brain evolved for sensorimotor
control, not classification. This is the smallest honest test of that: a 2D
light-seeking task where a controller sees 8 sensors and emits 2 motor commands.
It exists to answer three questions:

1. Can a fixed connectome plus a trained output layer control the agent at all?
2. Does it beat a trivial hand-written rule (turn toward the light)?
3. Does the connectome help beyond a direct linear map from the same 8 sensors
   to the same 2 motors? (This is the "does the graph matter" control - if a
   linear sensor->motor map matches it, the graph adds nothing, consistent with
   every classification result in docs/EXPERIMENTS.md.)

FROZEN INTERFACE - subagents must not change this file; import from it.

    from world import (Arena, SENSOR_DIM, ACTION_DIM, heuristic_action,
                       obs_vector, run_episode, run_many, summarize)

    env = Arena(seed=0)
    obs = env.reset()                  # np.ndarray shape (8,)
    obs, reward, done, info = env.step(np.array([turn, thrust]))
    # turn, thrust in [-1, 1]. info has 'dist' and 'success'.

    run_episode(controller, seed)      -> {'steps': int, 'success': bool, 'final_dist': float}
    run_many(controller, n, seed0)     -> aggregate dict (see summarize)
    # controller: callable(obs: np.ndarray) -> np.ndarray shape (2,) in [-1, 1]

Conventions that keep results comparable:
- Success = the agent comes within SUCCESS_RADIUS of the light within MAX_STEPS.
- Episodes are seeded; `run_many(fn, 200, seed0=1000)` is the standard evaluation.
- The heuristic teacher is the reference: a hand-written rule. Any learned
  controller should be reported as a gap against it, not in isolation.
"""

import math

import numpy as np

SENSOR_DIM = 7
ACTION_DIM = 2
MAX_STEPS = 200
ARENA_RADIUS = 1.0
SUCCESS_RADIUS = 0.08
TURN_RATE = 0.25          # radians per step at full command
THRUST_RATE = 0.06        # arena units per step at full command
START_MIN_DIST = 0.5      # agent starts at least this far from the light


def _norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (1.0, 0.0)


def obs_vector(pos, heading, light, visible=True):
    """The 7 sensors. Order is part of the frozen interface.

    FIXED 2026-09-12: this module previously declared SENSOR_DIM=8 while
    returning 7 values, which forced every lane-1 agent to slice a superfluous
    row. An (8, N) random draw and a (7, N) draw share identical rows 0-6, so
    the fix does not change any result already measured - verified directly.
    """
    to_light = (light[0] - pos[0], light[1] - pos[1])
    dist = math.hypot(*to_light)
    ux, uy = _norm(to_light)
    hx, hy = math.cos(heading), math.sin(heading)
    # bearing of the light relative to the agent's heading
    cross = hx * uy - hy * ux
    dot = hx * ux + hy * uy
    # distance to the arena wall behind/around the agent (proprioceptive hint)
    wall = ARENA_RADIUS - math.hypot(*pos)
    if visible:
        light_sensors = [dot, cross, 1.0 / (1.0 + 5.0 * dist), dist / (2 * ARENA_RADIUS)]
    else:
        light_sensors = [0.0, 0.0, 0.0, 0.0]
    return np.array(light_sensors + [hx, hy, wall], dtype=np.float64)


def heuristic_action(obs):
    """The hand-written reference rule: turn toward the light, thrust when aligned.

    obs[0] = cos(bearing), obs[1] = sin(bearing) = signed bearing. A positive
    bearing means the light lies counter-clockwise of the heading, so the turn
    command must be POSITIVE to close the gap (turning away gives 0% success -
    this sign is the difference between working and not).
    """
    cos_bearing, sin_bearing = obs[0], obs[1]
    turn = float(np.clip(sin_bearing * 2.0, -1.0, 1.0))
    thrust = float(np.clip(cos_bearing * 2.0, -1.0, 1.0))
    return np.array([turn, thrust], dtype=np.float64)


class MemoryHeuristic:
    """Hand-written memory rule for the partially observable mode.

    Keeps the last observed bearing and acts on it (persistence). A static light
    makes this a strong baseline: any recurrent controller must beat it to be
    worth anything, because it is one line of code with two variables of state.
    """

    def __init__(self, decay=1.0):
        self.decay = decay
        self.cos_b = 0.0
        self.sin_b = 0.0

    def __call__(self, obs):
        if abs(obs[0]) > 1e-9 or abs(obs[1]) > 1e-9:   # light currently visible
            self.cos_b, self.sin_b = float(obs[0]), float(obs[1])
        else:                                          # blind: rely on memory
            self.cos_b *= self.decay
            self.sin_b *= self.decay
        turn = float(np.clip(self.sin_b * 2.0, -1.0, 1.0))
        thrust = float(np.clip(self.cos_b * 2.0, -1.0, 1.0))
        return np.array([turn, thrust], dtype=np.float64)


class Arena:
    """A unit-disc arena with a point light source. Deterministic per seed.

    mode:
      "open"  - the light is always visible (fully observable).
      "blink" - the light sensors are zeroed except every `blink_period` steps,
                making the task partially observable and genuinely
                time-dependent. This is the mode where a recurrent substrate
                can plausibly beat a memoryless map.
    """

    def __init__(self, seed=0, mode="open", blink_period=3):
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.blink_period = blink_period
        self.pos = np.zeros(2)
        self.heading = 0.0
        self.light = np.zeros(2)
        self.steps = 0
        self.done = False
        self.reset()

    def light_visible(self):
        return self.mode == "open" or (self.steps % self.blink_period == 0)

    def _obs(self):
        return obs_vector(self.pos, self.heading, self.light, visible=self.light_visible())

    def reset(self):
        # light somewhere in the arena, distinct from the start position
        ang = self.rng.uniform(0, 2 * math.pi)
        rad = self.rng.uniform(0.0, 0.7 * ARENA_RADIUS)
        self.light = np.array([rad * math.cos(ang), rad * math.sin(ang)])
        p = np.zeros(2)
        for _ in range(100):
            p = self.rng.uniform(-0.85 * ARENA_RADIUS, 0.85 * ARENA_RADIUS, size=2)
            if math.hypot(*(p - self.light)) >= START_MIN_DIST:
                break
        self.pos = p
        self.heading = float(self.rng.uniform(0, 2 * math.pi))
        self.steps = 0
        self.done = False
        return self._obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
        turn, thrust = float(action[0]), float(action[1])
        self.heading += TURN_RATE * turn
        speed = THRUST_RATE * max(0.0, thrust)
        self.pos = self.pos + speed * np.array([math.cos(self.heading), math.sin(self.heading)])
        # stay inside the arena (slide along the wall)
        r = math.hypot(*self.pos)
        if r > ARENA_RADIUS:
            self.pos = self.pos * (ARENA_RADIUS / r)
        self.steps += 1
        dist = float(math.hypot(*(self.light - self.pos)))
        success = dist <= SUCCESS_RADIUS
        if success or self.steps >= MAX_STEPS:
            self.done = True
        reward = -dist + (10.0 if success else 0.0)
        return (self._obs(), reward, self.done,
                {"dist": dist, "success": success})


def run_episode(controller, seed=0, max_steps=MAX_STEPS, mode="open", blink_period=3):
    env = Arena(seed=seed, mode=mode, blink_period=blink_period)
    obs = env.reset()
    info = {"dist": float(math.hypot(*(env.light - env.pos))), "success": False}
    for _ in range(max_steps):
        obs, _reward, done, info = env.step(controller(obs))
        if done:
            break
    return {"steps": env.steps, "success": bool(info["success"]),
            "final_dist": float(info["dist"])}


def run_many(controller, n_episodes=200, seed0=1000, mode="open", blink_period=3):
    """Standard evaluation: fixed seed range so every controller sees the same arenas.

    A stateful controller (e.g. a reservoir) must be RESET before each episode;
    use `run_many_reset` for those, or pass a factory that builds a fresh one.
    """
    rows = [run_episode(controller, seed=seed0 + i, mode=mode, blink_period=blink_period)
            for i in range(n_episodes)]
    return summarize(rows)


def run_many_factory(factory, n_episodes=200, seed0=1000, mode="open", blink_period=3):
    """Same as run_many, but builds a fresh controller per episode (stateful ones)."""
    rows = [run_episode(factory(), seed=seed0 + i, mode=mode, blink_period=blink_period)
            for i in range(n_episodes)]
    return summarize(rows)


def summarize(rows):
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    return {
        "episodes": n,
        "success_rate": round(len(succ) / n, 4),
        "mean_steps_to_success": round(float(np.mean([r["steps"] for r in succ])), 1) if succ else None,
        "mean_final_dist": round(float(np.mean([r["final_dist"] for r in rows])), 4),
        "timeouts": n - len(succ),
    }


def random_action_factory(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=ACTION_DIM)


if __name__ == "__main__":
    import json
    for mode in ("open", "blink"):
        print(f"--- mode={mode} ---")
        print("heuristic      ", json.dumps(run_many(heuristic_action, 200, mode=mode)))
        print("memory-heurist ", json.dumps(run_many_factory(MemoryHeuristic, 200, mode=mode)))
        print("random         ", json.dumps(run_many(random_action_factory(0), 200, mode=mode)))
        print("stand-still    ", json.dumps(run_many(lambda obs: np.zeros(ACTION_DIM), 200, mode=mode)))

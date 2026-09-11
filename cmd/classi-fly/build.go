package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/caimlas/classi-fly/reservoir"
)

// buildConfig holds the `build` subcommand's inputs. build shells out to the
// offline Python tooling (tools/ingest/synthetic.py, then
// tools/train/train_readout.py) in a temp dir and packs the result: one
// command from a seed to a trained .fly. It requires Python on PATH; the
// shipping path is pack (the Python tooling is not part of the binary).
type buildConfig struct {
	synthetic   bool
	seed        int
	classesPath string
	pairsPath   string
	outPath     string
	neurons     int
	steps       int
	decay       float64
	embedDim    int
	reportPath  string
	pythonBin   string
	toolsDir    string
}

// runBuild implements `classi-fly build`.
func runBuild(args []string, stdout, stderr io.Writer) int {
	cfg := &buildConfig{}
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.BoolVar(&cfg.synthetic, "synthetic", false, "generate the reservoir via tools/ingest/synthetic.py (currently the only source)")
	fs.IntVar(&cfg.seed, "seed", 7, "synthetic reservoir seed")
	fs.StringVar(&cfg.classesPath, "classes", "", "JSON file holding the class-name array")
	fs.StringVar(&cfg.pairsPath, "pairs", "", "pairs.jsonl for training ({embedding,label} per line); when omitted a deterministic scaffold is generated from --classes")
	fs.StringVar(&cfg.outPath, "out", "", "output .fly path")
	fs.IntVar(&cfg.neurons, "neurons", 64, "synthetic reservoir neuron count")
	fs.IntVar(&cfg.steps, "steps", 8, "recurrence steps recorded in the artifact")
	fs.Float64Var(&cfg.decay, "decay", defaultDecay, "leak term written to the artifact header")
	fs.IntVar(&cfg.embedDim, "embed-dim", 8, "embedding dimension of the training pairs")
	fs.StringVar(&cfg.reportPath, "report", "", "optional readout_report.json path")
	fs.StringVar(&cfg.pythonBin, "python", "python3", "python interpreter used to run the offline tooling")
	fs.StringVar(&cfg.toolsDir, "tools-dir", "", "repo root holding tools/ingest and tools/train (default: beside the binary, then the working directory)")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	switch {
	case !cfg.synthetic:
		fmt.Fprintln(stderr, "classi-fly build: --synthetic is currently the only supported source")
		return exitUsage
	case cfg.outPath == "":
		fmt.Fprintln(stderr, "classi-fly build: --out is required")
		return exitUsage
	case cfg.classesPath == "":
		fmt.Fprintln(stderr, "classi-fly build: --classes is required")
		return exitUsage
	case cfg.neurons < 2:
		fmt.Fprintln(stderr, "classi-fly build: --neurons must be >= 2")
		return exitUsage
	case cfg.steps <= 0:
		fmt.Fprintln(stderr, "classi-fly build: --steps must be positive")
		return exitUsage
	case cfg.embedDim <= 0:
		fmt.Fprintln(stderr, "classi-fly build: --embed-dim must be positive")
		return exitUsage
	case cfg.decay <= 0 || cfg.decay >= 1:
		fmt.Fprintf(stderr, "classi-fly build: --decay must be in (0, 1), got %v\n", cfg.decay)
		return exitUsage
	}
	if err := buildRun(cfg, stdout, stderr); err != nil {
		fmt.Fprintf(stderr, "classi-fly build: %v\n", err)
		return exitUsage
	}
	return exitOK
}

// buildRun generates the adjacency, packs a provisional matrix-mode artifact
// (the trainer reads states from a .fly and cannot expand seed mode), trains
// the readout against it, then packs the final artifact with the trained
// readout. The provisional and final artifacts share the exact same graph,
// input projection, steps and decay, so the states the trainer saw are the
// states the runtime produces.
func buildRun(cfg *buildConfig, stdout, stderr io.Writer) error {
	if err := checkPython(cfg.pythonBin); err != nil {
		return err
	}
	toolsDir, err := findToolsDir(cfg.toolsDir)
	if err != nil {
		return err
	}
	classes, err := readClassesFile(cfg.classesPath)
	if err != nil {
		return err
	}

	tmp, err := os.MkdirTemp("", "classi-fly-build-")
	if err != nil {
		return fmt.Errorf("creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmp)

	if cfg.pairsPath == "" {
		// No pairs given: scaffold deterministic pairs from --classes so the
		// trainer has signal. Real deployments pass --pairs.
		pairs := filepath.Join(tmp, "pairs.jsonl")
		if err := writeScaffoldPairs(pairs, classes, cfg.embedDim); err != nil {
			return err
		}
		cfg.pairsPath = pairs
	}

	adjPath := filepath.Join(tmp, "adjacency.json")
	if err := runPythonTool(cfg.pythonBin,
		filepath.Join(toolsDir, "tools", "ingest", "synthetic.py"),
		[]string{"--seed", strconv.Itoa(cfg.seed), "--neurons", strconv.Itoa(cfg.neurons), "--out", adjPath},
		toolsDir, stdout, stderr); err != nil {
		return err
	}

	provPath := filepath.Join(tmp, "provisional.fly")
	pbPath := filepath.Join(tmp, "pack.json")
	if err := writeProvisionalPack(cfg, adjPath, provPath, pbPath); err != nil {
		return err
	}

	roPath := filepath.Join(tmp, "readout.json")
	trainArgs := []string{"--fly", provPath, "--pairs", cfg.pairsPath, "--out", roPath}
	if cfg.reportPath != "" {
		trainArgs = append(trainArgs, "--report", cfg.reportPath)
	}
	if err := runPythonTool(cfg.pythonBin,
		filepath.Join(toolsDir, "tools", "train", "train_readout.py"),
		trainArgs, toolsDir, stdout, stderr); err != nil {
		return err
	}

	var adj adjacencyExport
	if err := readJSONFile(adjPath, &adj); err != nil {
		return err
	}
	var ro readoutExport
	if err := readJSONFile(roPath, &ro); err != nil {
		return err
	}
	var pb packBlock
	if err := readJSONFile(pbPath, &pb); err != nil {
		return err
	}
	if err := packRun(adj, ro, pb, cfg.decay, false, cfg.outPath); err != nil {
		return err
	}

	// The deliverable must load: verify through the public API before
	// declaring success.
	r, err := reservoir.Load(cfg.outPath)
	if err != nil {
		return fmt.Errorf("verifying %s: %w", cfg.outPath, err)
	}
	info := r.Info()
	if info.Neurons != adj.Neurons || len(info.Classes) != len(ro.Classes) {
		return fmt.Errorf("verifying %s: got neurons=%d classes=%d, want neurons=%d classes=%d",
			cfg.outPath, info.Neurons, len(info.Classes), adj.Neurons, len(ro.Classes))
	}
	fmt.Fprintf(stdout, "wrote %s (neurons=%d, classes=%d, license=%s)\n",
		cfg.outPath, info.Neurons, len(info.Classes), info.License)
	return nil
}

// checkPython verifies the configured python interpreter runs at all and
// explains why build needs it when it does not.
func checkPython(bin string) error {
	if err := exec.Command(bin, "-c", "print('ok')").Run(); err != nil {
		return fmt.Errorf("python interpreter %q is not runnable: %w (the build subcommand shells out to tools/ingest/synthetic.py and tools/train/train_readout.py; pack, info and classify need no Python)", bin, err)
	}
	return nil
}

// runPythonTool runs pythonBin on script with scriptArgs, wiring the child's
// stdout/stderr through so tool errors surface verbatim.
func runPythonTool(bin, script string, scriptArgs []string, dir string, stdout, stderr io.Writer) error {
	argv := append([]string{script}, scriptArgs...)
	cmd := exec.Command(bin, argv...)
	cmd.Dir = dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", script, err)
	}
	return nil
}

// findToolsDir locates the repo root that holds tools/ingest/synthetic.py.
func findToolsDir(explicit string) (string, error) {
	var candidates []string
	if explicit != "" {
		candidates = append(candidates, explicit)
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, cwd)
	}
	for _, dir := range candidates {
		if dir == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, "tools", "ingest", "synthetic.py")); err == nil {
			return dir, nil
		}
	}
	return "", fmt.Errorf("cannot locate tools/ingest/synthetic.py; pass --tools-dir <repo root>")
}

// readClassesFile reads the class-name array JSON file.
func readClassesFile(path string) ([]string, error) {
	var classes []string
	if err := readJSONFile(path, &classes); err != nil {
		return nil, fmt.Errorf("reading classes: %w", err)
	}
	if len(classes) == 0 {
		return nil, fmt.Errorf("classes file %s holds no classes", path)
	}
	return classes, nil
}

// writeProvisionalPack packs the trainer's input artifact: the generated
// adjacency in matrix input mode (a deterministic per-class input lane,
// embedding-major as the container stores it), a placeholder zero readout
// the trainer overwrites, and the pack block the final pack reuses
// verbatim.
func writeProvisionalPack(cfg *buildConfig, adjPath, provPath, pbPath string) error {
	var adj adjacencyExport
	if err := readJSONFile(adjPath, &adj); err != nil {
		return fmt.Errorf("reading generated adjacency: %w", err)
	}
	pb := packBlock{
		EmbedDim:  cfg.embedDim,
		Steps:     cfg.steps,
		InputMode: "matrix",
		InScale:   0.1,
	}
	// One input lane per embedding dimension: in_w[j*N+i] couples
	// embedding j to neuron i, giving every class a distinct drive pattern
	// (the trainer only needs separable states). Byte 40 with in_scale 0.1
	// (unit lane drive ~4) produces strongly separated states through this
	// recurrence shape; sub-saturated drives leave states nearly collinear.
	pb.InW = make([]int8, cfg.embedDim*adj.Neurons)
	for c := 0; c < adj.Neurons; c++ {
		pb.InW[(c%cfg.embedDim)*adj.Neurons+c] = 40
	}
	pbBlob, err := json.Marshal(pb)
	if err != nil {
		return err
	}
	if err := os.WriteFile(pbPath, pbBlob, 0o644); err != nil {
		return err
	}

	ro := readoutExport{
		Classes:      []string{"provisional"},
		W:            make([][]float64, adj.Neurons),
		Bias:         []float64{0},
		Threshold:    []float64{0},
		ReadoutScale: 1,
	}
	for i := range ro.W {
		ro.W[i] = make([]float64, 1)
	}
	return packRun(adj, ro, pb, cfg.decay, false, provPath)
}

// writeScaffoldPairs writes a minimal deterministic pairs.jsonl so the
// trainer has signal when the caller has no real pairs yet. Each class ci
// sweeps the input lanes in rotation (lane (ci+variant) mod embedDim), a
// small mixing term on even variants — proven separable through the
// reservoir recurrence by the end-to-end test suite.
func writeScaffoldPairs(path string, classes []string, embedDim int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for variant := 0; variant < 6; variant++ {
		for ci, label := range classes {
			emb := make([]float64, embedDim)
			emb[(ci+variant)%embedDim] = 1
			if variant%2 == 0 {
				emb[(ci+1)%embedDim] = 0.5
			}
			if err := enc.Encode(map[string]any{"embedding": emb, "label": label}); err != nil {
				return err
			}
		}
	}
	return nil
}

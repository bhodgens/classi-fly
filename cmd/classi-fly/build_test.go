package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/caimlas/classi-fly/reservoir"
)

// repoRoot returns the repository root from the test's working directory
// (cmd/classi-fly), for locating tools/.
func repoRoot(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolving repo root: %v", err)
	}
	return abs
}

func TestBuildSyntheticEndToEnd(t *testing.T) {
	if err := checkPython("python3"); err != nil {
		t.Skip("python3 not available; build subcommand requires it")
	}
	dir := t.TempDir()
	classesPath := filepath.Join(dir, "classes.json")
	if err := writeJSONFileSync(classesPath, []string{"alpha", "beta", "gamma"}); err != nil {
		t.Fatalf("writing classes fixture: %v", err)
	}
	pairsPath := filepath.Join(dir, "pairs.jsonl")
	pf, err := os.Create(pairsPath)
	if err != nil {
		t.Fatalf("creating pairs fixture: %v", err)
	}
	embedDim := 4
	for variant := 0; variant < 6; variant++ {
		for ci, label := range []string{"alpha", "beta", "gamma"} {
			emb := make([]float64, embedDim)
			emb[(ci+variant)%embedDim] = 1
			if variant%2 == 0 {
				emb[(ci+1)%embedDim] = 0.5
			}
			if err := json.NewEncoder(pf).Encode(map[string]any{"embedding": emb, "label": label}); err != nil {
				pf.Close()
				t.Fatalf("writing pairs row: %v", err)
			}
		}
	}
	pf.Close()
	outPath := filepath.Join(dir, "built.fly")

	args := []string{
		"--synthetic",
		"--seed", "7",
		"--neurons", "24",
		"--steps", "4",
		"--embed-dim", "4",
		"--classes", classesPath,
		"--pairs", pairsPath,
		"--tools-dir", repoRoot(t),
		"--out", outPath,
	}
	if code := runBuild(args, &syncBuffer{}, &syncBuffer{}); code != exitOK {
		t.Fatalf("runBuild exit = %d, want %d", code, exitOK)
	}
	r, err := reservoir.Load(outPath)
	if err != nil {
		t.Fatalf("reservoir.Load on built artifact: %v", err)
	}
	info := r.Info()
	if info.Neurons != 24 {
		t.Errorf("built artifact neurons = %d, want 24", info.Neurons)
	}
	if len(info.Classes) != 3 {
		t.Errorf("built artifact classes = %v, want 3 entries", info.Classes)
	}
	if info.License != "none" {
		t.Errorf("built artifact license = %q, want %q", info.License, "none")
	}
}

// TestBuildScaffoldPairs exercises the --pairs-omitted path so the command
// works from classes alone.
func TestBuildScaffoldPairs(t *testing.T) {
	if err := checkPython("python3"); err != nil {
		t.Skip("python3 not available; build subcommand requires it")
	}
	dir := t.TempDir()
	classesPath := filepath.Join(dir, "classes.json")
	if err := writeJSONFileSync(classesPath, []string{"x", "y"}); err != nil {
		t.Fatalf("writing classes fixture: %v", err)
	}
	outPath := filepath.Join(dir, "built.fly")
	args := []string{
		"--synthetic",
		"--seed", "3",
		"--neurons", "16",
		"--steps", "2",
		"--embed-dim", "4",
		"--classes", classesPath,
		"--tools-dir", repoRoot(t),
		"--out", outPath,
	}
	if code := runBuild(args, &syncBuffer{}, &syncBuffer{}); code != exitOK {
		t.Fatalf("runBuild exit = %d, want %d", code, exitOK)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("build did not write %s: %v", outPath, err)
	}
}

// TestBuildMissingPython verifies the clear-error exit-1 contract when the
// Python tooling cannot run.
func TestBuildMissingPython(t *testing.T) {
	dir := t.TempDir()
	classesPath := filepath.Join(dir, "classes.json")
	if err := writeJSONFileSync(classesPath, []string{"x"}); err != nil {
		t.Fatalf("writing classes fixture: %v", err)
	}
	args := []string{
		"--synthetic",
		"--classes", classesPath,
		"--tools-dir", repoRoot(t),
		"--python", "classi-fly-definitely-not-a-binary",
		"--out", filepath.Join(dir, "built.fly"),
	}
	if code := runBuild(args, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runBuild with missing python exit = %d, want %d", code, exitUsage)
	}
}

func TestBuildUsage(t *testing.T) {
	if code := runBuild(nil, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runBuild(nil) exit = %d, want %d", code, exitUsage)
	}
	dir := t.TempDir()
	classesPath := filepath.Join(dir, "classes.json")
	if err := writeJSONFileSync(classesPath, []string{"x"}); err != nil {
		t.Fatalf("writing classes fixture: %v", err)
	}
	// Missing --out: usage error.
	if code := runBuild([]string{"--synthetic", "--classes", classesPath}, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runBuild without --out exit = %d, want %d", code, exitUsage)
	}
}

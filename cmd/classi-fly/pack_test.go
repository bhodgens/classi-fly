package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/caimlas/classi-fly/reservoir"
)

// writePackFixtures writes a tiny deterministic adjacency + readout export
// pair (Contract 3 shapes) into dir and returns their paths. n neurons, k
// classes, matching dimensions.
func writePackFixtures(t *testing.T, dir, license string, n, k int) (adjPath, roPath, pbPath string) {
	t.Helper()
	indptr := make([]uint32, n+1)
	indices := make([]uint32, 0, n*2)
	weights := make([]float64, 0, n*2)
	for row := 0; row < n; row++ {
		for j := 1; j <= 2; j++ {
			indices = append(indices, uint32((row+j)%n))
			weights = append(weights, float64(row+j)*0.1)
		}
		indptr[row+1] = uint32(len(indices))
	}
	adj := adjacencyExport{
		Name:        "test-reservoir",
		Neurons:     n,
		Edges:       len(indices),
		Indptr:      indptr,
		Indices:     indices,
		Weights:     weights,
		Source:      "synthetic",
		License:     license,
		Attribution: "test fixture",
	}
	ro := readoutExport{Classes: make([]string, k)}
	for c := 0; c < k; c++ {
		ro.Classes[c] = "class" + string(rune('a'+c))
	}
	ro.W = make([][]float64, n)
	for i := range ro.W {
		row := make([]float64, k)
		for c := range row {
			row[c] = float64(i+c) * 0.05
		}
		ro.W[i] = row
	}
	ro.Bias = make([]float64, k)
	ro.Threshold = make([]float64, k)
	for c := range ro.Bias {
		ro.Bias[c] = 0.01
		ro.Threshold[c] = 0.3
	}
	ro.WeightScale = 1
	ro.ReadoutScale = 1

	pb := packBlock{
		EmbedDim:  2,
		Steps:     4,
		InputMode: "matrix",
		InW:       make([]int8, 2*n),
		InScale:   0.1,
	}
	for i := range pb.InW {
		pb.InW[i] = int8((i % 7) - 3)
	}

	adjPath = filepath.Join(dir, "adjacency.json")
	roPath = filepath.Join(dir, "readout.json")
	pbPath = filepath.Join(dir, "pack.json")
	if err := writeJSONFileSync(adjPath, adj); err != nil {
		t.Fatalf("writing adjacency fixture: %v", err)
	}
	if err := writeJSONFileSync(roPath, ro); err != nil {
		t.Fatalf("writing readout fixture: %v", err)
	}
	if err := writeJSONFileSync(pbPath, pb); err != nil {
		t.Fatalf("writing pack fixture: %v", err)
	}
	return adjPath, roPath, pbPath
}

func TestPackRoundTrip(t *testing.T) {
	dir := t.TempDir()
	adjPath, roPath, pbPath := writePackFixtures(t, dir, "none", 4, 2)
	outPath := filepath.Join(dir, "model.fly")

	code := runPack([]string{
		"--adjacency", adjPath,
		"--readout", roPath,
		"--pack", pbPath,
		"--out", outPath,
	}, os.Stdout, os.Stderr)
	if code != exitOK {
		t.Fatalf("runPack exit = %d, want %d", code, exitOK)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("pack did not write %s: %v", outPath, err)
	}

	r, err := reservoir.Load(outPath)
	if err != nil {
		t.Fatalf("reservoir.Load: %v", err)
	}
	info := r.Info()
	if info.Neurons != 4 {
		t.Errorf("Info().Neurons = %d, want 4", info.Neurons)
	}
	if info.Edges != 8 {
		t.Errorf("Info().Edges = %d, want 8", info.Edges)
	}
	if info.EmbedDim != 2 {
		t.Errorf("Info().EmbedDim = %d, want 2", info.EmbedDim)
	}
	if info.Steps != 4 {
		t.Errorf("Info().Steps = %d, want 4", info.Steps)
	}
	if len(info.Classes) != 2 || info.Classes[0] != "classa" {
		t.Errorf("Info().Classes = %v, want [classa classb]", info.Classes)
	}
	if info.License != "none" {
		t.Errorf("Info().License = %q, want %q", info.License, "none")
	}
}

func TestPackLicenseGate(t *testing.T) {
	dir := t.TempDir()
	adjPath, roPath, pbPath := writePackFixtures(t, dir, "CC-BY-NC-4.0", 4, 2)
	outPath := filepath.Join(dir, "model.fly")

	args := []string{
		"--adjacency", adjPath,
		"--readout", roPath,
		"--pack", pbPath,
		"--out", outPath,
	}
	if code := runPack(args, os.Stdout, os.Stderr); code != exitLicense {
		t.Fatalf("runPack without --allow-noncommercial exit = %d, want %d", code, exitLicense)
	}
	if code := runPack(append(args, "--allow-noncommercial"), os.Stdout, os.Stderr); code != exitOK {
		t.Fatalf("runPack --allow-noncommercial exit = %d, want %d", code, exitOK)
	}
}

func TestPackDimensionMismatch(t *testing.T) {
	dir := t.TempDir()
	adjPath, roPath, pbPath := writePackFixtures(t, dir, "none", 4, 2)
	outPath := filepath.Join(dir, "model.fly")

	// Shrink W to 3 rows: must disagree with the adjacency's 4 neurons.
	ro := readoutExport{}
	if err := readJSONFile(roPath, &ro); err != nil {
		t.Fatalf("reading fixture: %v", err)
	}
	ro.W = ro.W[:3]
	badPath := filepath.Join(dir, "readout_bad.json")
	if err := writeJSONFileSync(badPath, ro); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}

	code := runPack([]string{
		"--adjacency", adjPath,
		"--readout", badPath,
		"--pack", pbPath,
		"--out", outPath,
	}, os.Stdout, os.Stderr)
	if code != exitLoad {
		t.Fatalf("runPack exit = %d, want %d", code, exitLoad)
	}
}

func TestPackUsage(t *testing.T) {
	if code := runPack(nil, os.Stdout, os.Stderr); code != exitUsage {
		t.Fatalf("runPack(nil) exit = %d, want %d", code, exitUsage)
	}
}

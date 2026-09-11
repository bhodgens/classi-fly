package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeJSONFileSync marshals v and writes it to path.
func writeJSONFileSync(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// writeFixtureFly writes a minimal .fly framing (magic + header + placeholder
// payload) so framing-only consumers (info) can be exercised without the
// full reservoir package. The payload is junk; only Load cares about it.
func writeFixtureFly(t *testing.T, path string, header headerLite) {
	t.Helper()
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshaling fixture header: %v", err)
	}
	fly := append([]byte(flyMagic), byte(uint32(len(raw))), byte(uint32(len(raw))>>8), byte(uint32(len(raw))>>16), byte(uint32(len(raw))>>24))
	fly = append(fly, raw...)
	fly = append(fly, 0, 0, 0, 0) // placeholder payload bytes
	if err := os.WriteFile(path, fly, 0o644); err != nil {
		t.Fatalf("writing fixture .fly: %v", err)
	}
}

func TestInfoVerbatimHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.fly")
	header := headerLite{
		Format: "fly-reservoir", Version: 1, Name: "fixture",
		Neurons: 4, Edges: 8, EmbedDim: 2, Steps: 4,
		Classes:     []string{"a", "b"},
		License:     "none",
		Attribution: "test",
	}
	writeFixtureFly(t, path, header)

	stdout := &syncBuffer{}
	if code := runInfo([]string{path}, stdout, &syncBuffer{}); code != exitOK {
		t.Fatalf("runInfo exit = %d, want %d", code, exitOK)
	}
	out := stdout.String()
	if len(out) == 0 || out[len(out)-1] != '\n' {
		t.Errorf("info output not newline-terminated: %q", out)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(out), &decoded); err != nil {
		t.Fatalf("info output is not JSON: %v (%q)", err, out)
	}
	if decoded["format"] != "fly-reservoir" {
		t.Errorf(`info output format = %v, want "fly-reservoir"`, decoded["format"])
	}
	// Verbatim check: the printed bytes (minus the trailing newline) must
	// equal the header JSON stored in the file.
	raw, _, err := readFlyHeader(path)
	if err != nil {
		t.Fatalf("readFlyHeader: %v", err)
	}
	if got := out[:len(out)-1]; got != string(raw) {
		t.Errorf("info output is not verbatim:\n got %q\nwant %q", got, string(raw))
	}
}

func TestInfoLicenseGate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.fly")
	writeFixtureFly(t, path, headerLite{
		Format: "fly-reservoir", Version: 1, Name: "nc",
		Neurons: 4, Edges: 8, EmbedDim: 2, Steps: 4,
		Classes: []string{"a"}, License: "CC-BY-NC-4.0", Attribution: "flywire",
	})
	if code := runInfo([]string{path}, &syncBuffer{}, &syncBuffer{}); code != exitLicense {
		t.Fatalf("runInfo exit = %d, want %d", code, exitLicense)
	}
	if code := runInfo([]string{path, "--allow-noncommercial"}, &syncBuffer{}, &syncBuffer{}); code != exitOK {
		t.Fatalf("runInfo --allow-noncommercial exit = %d, want %d", code, exitOK)
	}
}

func TestInfoUsageAndMissing(t *testing.T) {
	if code := runInfo(nil, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runInfo(nil) exit = %d, want %d", code, exitUsage)
	}
	if code := runInfo([]string{filepath.Join(t.TempDir(), "nope.fly")}, &syncBuffer{}, &syncBuffer{}); code != exitLoad {
		t.Fatalf("runInfo(missing) exit = %d, want %d", code, exitLoad)
	}
}

func TestClassifyOutputKeys(t *testing.T) {
	dir := t.TempDir()
	flyPath := filepath.Join(dir, "model.fly")
	adjPath, roPath, pbPath := writePackFixtures(t, dir, "none", 4, 2)
	if code := runPack([]string{
		"--adjacency", adjPath, "--readout", roPath, "--pack", pbPath, "--out", flyPath,
	}, &syncBuffer{}, &syncBuffer{}); code != exitOK {
		t.Fatalf("runPack exit = %d, want %d", code, exitOK)
	}

	stdout := &syncBuffer{}
	if code := runClassify([]string{flyPath, "--embedding", "0.5,0.25"}, stdout, &syncBuffer{}); code != exitOK {
		t.Fatalf("runClassify exit = %d, want %d", code, exitOK)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(stdout.String()), &decoded); err != nil {
		t.Fatalf("classify output is not JSON: %v (%q)", err, stdout.String())
	}
	for _, key := range []string{"class", "confidence", "margin", "abstained"} {
		if _, ok := decoded[key]; !ok {
			t.Errorf("classify output missing key %q: %v", key, decoded)
		}
	}
}

func TestClassifyWrongLengthExitsTwo(t *testing.T) {
	dir := t.TempDir()
	flyPath := filepath.Join(dir, "model.fly")
	adjPath, roPath, pbPath := writePackFixtures(t, dir, "none", 4, 2)
	if code := runPack([]string{
		"--adjacency", adjPath, "--readout", roPath, "--pack", pbPath, "--out", flyPath,
	}, &syncBuffer{}, &syncBuffer{}); code != exitOK {
		t.Fatalf("runPack exit = %d, want %d", code, exitOK)
	}
	// Embedding dim is 2; a 1-value embedding must exit 2 (dimension error).
	if code := runClassify([]string{flyPath, "--embedding", "0.5"}, &syncBuffer{}, &syncBuffer{}); code != exitLoad {
		t.Fatalf("runClassify wrong-length exit = %d, want %d", code, exitLoad)
	}
}

func TestClassifyUsage(t *testing.T) {
	if code := runClassify(nil, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runClassify(nil) exit = %d, want %d", code, exitUsage)
	}
	// Neither nor both embedding flags: usage error.
	dir := t.TempDir()
	embPath := filepath.Join(dir, "emb.json")
	if err := writeJSONFileSync(embPath, []float64{0.1, 0.2}); err != nil {
		t.Fatalf("writing embedding fixture: %v", err)
	}
	if code := runClassify([]string{"x.fly", "--embedding", "0.1", "--embedding-file", embPath}, &syncBuffer{}, &syncBuffer{}); code != exitUsage {
		t.Fatalf("runClassify both-flags exit = %d, want %d", code, exitUsage)
	}
}

func TestParseEmbedding(t *testing.T) {
	got, err := parseEmbedding(" 0.1, -2 ,3e-1 ")
	if err != nil {
		t.Fatalf("parseEmbedding: %v", err)
	}
	want := []float32{0.1, -2, 0.3}
	if len(got) != len(want) {
		t.Fatalf("parseEmbedding len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseEmbedding[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if _, err := parseEmbedding("0.1,,2"); err == nil {
		t.Error("parseEmbedding accepted an empty value")
	}
	if _, err := parseEmbedding("0.1,abc"); err == nil {
		t.Error("parseEmbedding accepted a non-number")
	}
}

func TestReadEmbeddingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "emb.json")
	if err := writeJSONFileSync(path, []float64{1, 2.5, -3}); err != nil {
		t.Fatalf("writing embedding: %v", err)
	}
	got, err := readEmbeddingFile(path)
	if err != nil {
		t.Fatalf("readEmbeddingFile: %v", err)
	}
	if len(got) != 3 || got[0] != 1 || got[1] != 2.5 || got[2] != -3 {
		t.Errorf("readEmbeddingFile = %v, want [1 2.5 -3]", got)
	}
}

// syncBuffer is an io.Writer collecting bytes for assertions.
type syncBuffer struct {
	b []byte
}

func (w *syncBuffer) Write(p []byte) (int, error) {
	w.b = append(w.b, p...)
	return len(p), nil
}

func (w *syncBuffer) String() string { return string(w.b) }

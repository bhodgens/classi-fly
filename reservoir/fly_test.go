package reservoir

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

// fixtureArtifact builds a tiny but non-trivial matrix-mode artifact: 8
// neurons, 3 classes, a diagonal-plus-coupling CSR graph, a permutation-
// structured input projection, and a readout with one dominant column per
// class. Deterministic: the PRNG is seeded, never global.
func fixtureArtifact(t *testing.T) Artifact {
	t.Helper()
	const n, d, k = 8, 6, 3

	w := make([]float64, 0, 3*n)
	indices := make([]uint32, 0, 3*n)
	indptr := make([]uint32, n+1)
	for i := 0; i < n; i++ {
		indptr[i] = uint32(len(indices))
		indices = append(indices, uint32(i))
		w = append(w, 0.8+0.05*float64(i%3))
		indices = append(indices, uint32((i+1)%n))
		w = append(w, 0.3)
		indices = append(indices, uint32((i+4)%n))
		w = append(w, 0.2)
	}
	indptr[n] = uint32(len(indices))

	inW := make([]float64, n*d)
	for j := 0; j < n && j < d; j++ {
		inW[j*n+j%n] = 1.0 // dominant lane per embedding dim
	}

	readout := make([]float64, n*k)
	for c := 0; c < k; c++ {
		readout[c*k+c] = 12
		readout[((c+1)%k)*k+c] = 2
	}
	// Extra embedding dims beyond the one-hot lanes must not inject noise
	// into the dominant neuron: the noise fill above only covers non-struct
	// positions, so zero the projections from dims >= n explicitly (a
	// "silent tail" mirrors real packers, where embed_dim may exceed the
	// lane count).
	for j := n; j < d; j++ {
		for i := 0; i < n; i++ {
			inW[j*n+i] = 0
		}
	}

	a := Artifact{
		Info: Info{
			Name: "fixture", Neurons: n, Edges: len(indices), EmbedDim: d,
			Steps: 4, Classes: []string{"alpha", "beta", "gamma"},
			License: "CC0-1.0", Attribution: "test fixture",
		},
		WeightScale:  0.05,
		InScale:      0.5,
		ReadoutScale: 0.1,
		Decay:        0.8,
		Seed:         7,
		Indptr:       indptr,
		Indices:      indices,
		Weights:      w,
		InW:          inW,
		Readout:      readout,
		Bias:         []float64{0, 0.1, -0.1},
		Threshold:    []float64{0.1, 0.1, 0.1},
		Source:       "synthetic",
		CreatedUTC:   "2026-09-11T00:00:00Z",
	}
	return a
}

func writeFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.fly")
	if err := WriteFile(path, fixtureArtifact(t)); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	return path
}

func TestLoad_MissingFileFailsClosed(t *testing.T) {
	if _, err := Load("/no/such/file.fly"); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoad_CorruptMagicFailsClosed(t *testing.T) {
	path := writeFixture(t)
	b, err := osReadAll(path)
	if err != nil {
		t.Fatal(err)
	}
	b[3] = 'X'
	corrupt := filepath.Join(t.TempDir(), "corrupt.fly")
	if err := osWriteFile(corrupt, b); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(corrupt); err == nil {
		t.Fatal("expected error for corrupt magic")
	}
}

func TestLoad_TruncatedFailsClosed(t *testing.T) {
	path := writeFixture(t)
	b, err := osReadAll(path)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := filepath.Join(t.TempDir(), "truncated.fly")
	if err := osWriteFile(corrupt, b[:len(b)/2]); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(corrupt); err == nil {
		t.Fatal("expected error for truncated file")
	}
}

func TestLoad_RoundTripDims(t *testing.T) {
	path := writeFixture(t)
	r, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	info := r.Info()
	want := fixtureArtifact(t)
	if info.Name != want.Info.Name || info.Neurons != want.Info.Neurons ||
		info.Edges != want.Info.Edges || info.EmbedDim != want.Info.EmbedDim ||
		info.Steps != want.Info.Steps || info.License != want.Info.License ||
		info.Attribution != want.Info.Attribution {
		t.Fatalf("Info mismatch: %+v", info)
	}
	if len(info.Classes) != len(want.Info.Classes) {
		t.Fatalf("Classes mismatch: %v", info.Classes)
	}
	for i := range info.Classes {
		if info.Classes[i] != want.Info.Classes[i] {
			t.Fatalf("Classes[%d] = %q want %q", i, info.Classes[i], want.Info.Classes[i])
		}
	}
}

func TestLoad_DeterministicReload(t *testing.T) {
	path := writeFixture(t)
	r1, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	h1, h2 := matrixChecksum(r1), matrixChecksum(r2)
	if h1 != h2 {
		t.Fatalf("reload checksum differs: %v vs %v", h1, h2)
	}
	// A third load through a fresh path must still agree.
	path3 := filepath.Join(t.TempDir(), "again.fly")
	if err := WriteFile(path3, fixtureArtifact(t)); err != nil {
		t.Fatal(err)
	}
	r3, err := Load(path3)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != matrixChecksum(r3) {
		t.Fatal("checksum differs across writes")
	}
}

func TestClassify_MatchesPythonReference(t *testing.T) {
	path := writeFixture(t)
	r, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	// One-hot embeddings excite exactly one lane; the python trainer
	// (tools/train/states.py + flybytes.py semantics) classifies the same
	// way, so this doubles as a cross-language sanity anchor.
	emb := make([]float32, 6)
	emb[0] = 10
	res, err := r.Classify(emb)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if res.Class != "alpha" {
		t.Fatalf("class = %q, want alpha (result %+v)", res.Class, res)
	}
	if res.Confidence <= 0.3 || res.Confidence > 1 {
		t.Fatalf("confidence %v out of range", res.Confidence)
	}
	if res.Abstained {
		t.Fatal("strong signal must not abstain")
	}
}

func TestClassify_AbstainsBelowThreshold(t *testing.T) {
	a := fixtureArtifact(t)
	a.Threshold = []float64{0.99, 0.99, 0.99}
	path := filepath.Join(t.TempDir(), "high-thresh.fly")
	if err := WriteFile(path, a); err != nil {
		t.Fatal(err)
	}
	r, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	emb := make([]float32, 6)
	emb[1] = 4
	res, err := r.Classify(emb)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Abstained {
		t.Fatalf("expected abstain below 0.99 threshold, got %+v", res)
	}
	if res.Confidence != 0 || res.Margin != 0 {
		t.Fatalf("abstained result must carry zero confidence/margin: %+v", res)
	}
	if res.Class != "beta" {
		t.Fatalf("abstained result should still name the top class, got %q", res.Class)
	}
}

func TestClassify_DimensionMismatchIsError(t *testing.T) {
	path := writeFixture(t)
	r, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Classify(make([]float32, 5)); err == nil {
		t.Fatal("expected dimension mismatch error")
	}
	if _, err := r.Classify(nil); err == nil {
		t.Fatal("expected error for nil embedding")
	}
}

func TestClassify_DeterministicUnderRepeat(t *testing.T) {
	path := writeFixture(t)
	r, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(9))
	for iter := 0; iter < 50; iter++ {
		emb := make([]float32, 6)
		for i := range emb {
			emb[i] = float32(rng.NormFloat64())
		}
		first, err := r.Classify(emb)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		for rep := 0; rep < 5; rep++ {
			again, err := r.Classify(emb)
			if err != nil {
				t.Fatalf("iter %d rep %d: %v", iter, rep, err)
			}
			if again != first {
				t.Fatalf("iter %d: nondeterministic classify: %+v vs %+v", iter, first, again)
			}
		}
	}
}

func TestQuantization_WithinDocumentedTolerance(t *testing.T) {
	// int8 quantization with scale s keeps each dequantized element within
	// s/2 of the original float (rounding is to nearest, ties to even); the
	// documented guarantee in WriteFile is |dequant(w) - w| <= scale/2.
	const n, d = 8, 6
	a := fixtureArtifact(t)
	scale := a.WeightScale
	// Simulate the quantization WriteFile performs, element by element.
	check := func(orig, quant []float64, s float64, what string) {
		t.Helper()
		for i := range orig {
			q := quantizeToInt8(orig[i] / s)
			back := float64(q) * s
			if diff := math.Abs(back - orig[i]); diff > s/2+1e-12 {
				t.Fatalf("%s[%d]: |%v - %v| = %v exceeds scale/2 (%v)", what, i, back, orig[i], diff, s/2)
			}
			if int32(q) != int32(quantizeToInt8(quant[i]/s)) {
				t.Fatalf("%s[%d]: quantized value drift", what, i)
			}
		}
	}
	quant := make([]float64, len(a.Weights))
	for i, v := range a.Weights {
		quant[i] = v
	}
	check(a.Weights, quant, scale, "weights")

	inWQ := make([]float64, len(a.InW))
	for i, v := range a.InW {
		inWQ[i] = v
	}
	check(a.InW, inWQ, a.InScale, "in_w")

	readoutQ := make([]float64, len(a.Readout))
	for i, v := range a.Readout {
		readoutQ[i] = v
	}
	check(a.Readout, readoutQ, a.ReadoutScale, "readout")
	_ = n
	_ = d
}

func TestLoad_SeedModeRoundTrip(t *testing.T) {
	a := fixtureArtifact(t)
	a.InW = nil
	a.InputMode = InputModeSeed
	path := filepath.Join(t.TempDir(), "seed.fly")
	if err := WriteFile(path, a); err != nil {
		t.Fatal(err)
	}
	r, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	info := r.Info()
	if info.Neurons != 8 || info.Edges != 24 {
		t.Fatalf("unexpected dims: %+v", info)
	}
	emb := make([]float32, 6)
	emb[2] = 5
	res, err := r.Classify(emb)
	if err != nil {
		t.Fatal(err)
	}
	if res.Class == "" {
		t.Fatal("expected a class name")
	}
}

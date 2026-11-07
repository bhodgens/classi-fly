package reservoir

// Property-based tests for Reservoir.Classify scoring/ranking
// (classify.go:70-167), hand-rolled with a deterministic seeded PRNG and
// zero new dependencies.
//
// Targets and properties (file:line refs to the implementation contract):
//
//	Target Reservoir.Classify over randomized valid artifacts
//	(buildArtifactProperty / buildAndLoadProperty):
//	  P1 probability simplex: when Classify does NOT abstain, the winning
//	     Confidence lies in (0, 1] and Margin in [0, 1]; when it abstains,
//	     Confidence and Margin are exactly 0 and Class still names a
//	     declared class (classify.go:26-31 contract; 163-166 impl).
//	  P2 dimension contract: an embedding of the wrong length is always an
//	     error and never a valid Result (classify.go:74-77).
//	P3 hostile embeddings are handled, never panic: an embedding
//	   containing NaN/±Inf is rejected up front by the non-finite guard
//	   (classify.go) and must yield an abstained Result with exact-zero
//	   Confidence/Margin; extreme-magnitude finite float32 values go
//	   through the full pipeline and, when non-abstained, must satisfy
//	   P1's bounds. (FIXED FINDING 1: before the guard, a NaN embedding
//	   produced a non-abstained Result with Confidence=NaN because
//	   NaN < threshold is always false, so the abstain gate could not
//	   fire. The assertion below pins the fixed behavior.)
//	  P4 determinism: identical (reservoir, embedding) pairs produce
//	     bit-identical Results (classify.go:68-69 "fixed iteration order,
//	     no map iteration" contract).
//	  P5 threshold semantics: Confidence < threshold of the winning class
//	     IMPLIES Abstained==true, and Abstained==true implies zero
//	     Confidence/Margin (classify.go:163-165). Verified against the
//	     artifact's own thresholds.
//	  P6 nil receiver: Classify on a nil *Reservoir is an error, never a
//	     panic (classify.go:71-73).
import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

// buildArtifactProperty builds a randomized valid Artifact (validateArtifact-
// clean): random CSR, either seed- or matrix-mode input, random readout,
// bias, and thresholds in [0, 0.9].
func buildArtifactProperty(rng *rand.Rand) Artifact {
	m, _, _ := randCSR(rng)
	n, k, d := m.n, 2+rng.Intn(3), 1+rng.Intn(4)
	wScale := 0.01 + rng.Float64()*0.5
	rScale := 0.01 + rng.Float64()*0.5
	a := Artifact{
		Info: Info{
			Name: "prop", Neurons: n, Edges: len(m.indices), EmbedDim: d,
			Steps: 1 + rng.Intn(8),
			Classes: func() []string {
				cs := []string{"alpha", "beta", "gamma", "delta", "epsilon"}
				return cs[:k]
			}(),
			License: "CC0-1.0", Attribution: "property test",
		},
		WeightScale:  wScale,
		InScale:      0.1 + rng.Float64(),
		ReadoutScale: rScale,
		Decay:        0.05 + rng.Float64()*0.9,
		Seed:         rng.Uint64(),
		Indptr:       m.indptr,
		Indices:      m.indices,
		// Weights must stay inside the int8 quantization envelope
		// (|v| <= 126*scale, same scale draw as the field above) so
		// WriteFile's checked quantizer accepts them.
		Weights: randWithin(rng, len(m.vals), 126*wScale),
		Readout: randWithin(rng, n*k, 126*rScale),
		Bias: func() []float64 {
			b := make([]float64, k)
			for i := range b {
				b[i] = rng.Float64()*2 - 1
			}
			return b
		}(),
		Threshold: func() []float64 {
			th := make([]float64, k)
			for i := range th {
				th[i] = rng.Float64() * 0.9 // deliberately below 1 so non-abstain is reachable
			}
			return th
		}(),
	}
	if rng.Intn(2) == 0 { // matrix mode half the time
		a.InW = func() []float64 {
			w := make([]float64, d*n)
			for i := range w {
				w[i] = float64(rng.Intn(5) - 2)
			}
			return w
		}()
	}
	return a
}

// buildAndLoadProperty writes the artifact and loads it back as a live
// Reservoir.
func buildAndLoadProperty(t *testing.T, rng *rand.Rand) (*Reservoir, Artifact) {
	t.Helper()
	a := buildArtifactProperty(rng)
	path := filepath.Join(t.TempDir(), "prop.fly")
	if err := WriteFile(path, a); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	r, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return r, a
}

// randEmbedding returns a float32 embedding of length d, mixing zeros,
// normals, and large magnitudes.
func randEmbedding(rng *rand.Rand, d int) []float32 {
	x := make([]float32, d)
	for i := range x {
		switch rng.Intn(5) {
		case 0:
			x[i] = 0
		case 1:
			x[i] = float32(rng.Float64() * 1e12)
		case 2:
			x[i] = -float32(rng.Float64() * 1e12)
		default:
			x[i] = float32(rng.Float64()*4 - 2)
		}
	}
	return x
}

// checkShape asserts the P1/P5 result-shape invariants for one outcome.
func checkShape(t *testing.T, seed string, c int, res Result, a *Artifact) {
	t.Helper()
	if res.Abstained {
		if res.Confidence != 0 || res.Margin != 0 {
			t.Fatalf("%s case=%d: abstained result carries Confidence=%g Margin=%g (want exact 0)",
				seed, c, res.Confidence, res.Margin)
		}
		return
	}
	if !(res.Confidence > 0 && res.Confidence <= 1) {
		t.Fatalf("%s case=%d: non-abstained Confidence=%g outside (0,1]", seed, c, res.Confidence)
	}
	if res.Margin < 0 || res.Margin > 1 {
		t.Fatalf("%s case=%d: non-abstained Margin=%g outside [0,1]", seed, c, res.Margin)
	}
	if a != nil {
		ci := -1
		for i, cl := range a.Info.Classes {
			if cl == res.Class {
				ci = i
				break
			}
		}
		if ci < 0 {
			t.Fatalf("%s case=%d: Class=%q not in artifact classes %v", seed, c, res.Class, a.Info.Classes)
		}
		if res.Confidence < a.Threshold[ci] {
			t.Fatalf("%s case=%d: Confidence=%g < threshold[%d]=%g but did not abstain",
				seed, c, res.Confidence, ci, a.Threshold[ci])
		}
	}
}

// TestPropertyClassify_SimplexAndThreshold is P1+P5.
func TestPropertyClassify_SimplexAndThreshold(t *testing.T) {
	const seed = 20261112
	rng := rand.New(rand.NewSource(seed))
	abstained := 0
	for c := 0; c < 600; c++ {
		r, a := buildAndLoadProperty(t, rng)
		x := randEmbedding(rng, a.Info.EmbedDim)
		res, err := r.Classify(x)
		if err != nil {
			t.Fatalf("case=%d: Classify: %v", c, err)
		}
		if res.Abstained {
			abstained++
		}
		checkShape(t, "20261112", c, res, &a)
	}
	if abstained == 0 {
		t.Fatalf("seed=%d: no case ever abstained; generator cannot reach the abstain path", seed)
	}
}

// TestPropertyClassify_WrongDimErrors is P2.
func TestPropertyClassify_WrongDimErrors(t *testing.T) {
	const seed = 20261113
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 200; c++ {
		r, a := buildAndLoadProperty(t, rng)
		d := a.Info.EmbedDim
		for _, wrong := range []int{d + 1, d - 1, 0, d + 7} {
			if wrong < 0 {
				continue
			}
			res, err := r.Classify(make([]float32, wrong))
			if err == nil {
				t.Fatalf("seed=%d case=%d: dim %d (artifact %d) accepted: %+v",
					seed, c, wrong, d, res)
			}
		}
	}
}

// TestPropertyClassify_HostileEmbeddingsNoPanic is P3. FINDING 1 (fixed):
// an embedding containing NaN used to produce a non-abstained Result with
// Confidence=NaN (NaN < threshold is false at the abstain gate, so it
// could not fire). The non-finite guard in Classify now rejects NaN/±Inf
// embeddings up front as an abstained result, so this test asserts the
// fixed behavior: every poison-free-hostile case that produced NaN used to
// be logged as a witness; now any non-finite value in the input MUST
// yield Abstained=true with exact-zero Confidence/Margin, and everything
// else must satisfy the simplex bounds.
func TestPropertyClassify_HostileEmbeddingsNoPanic(t *testing.T) {
	const seed = 20261114
	rng := rand.New(rand.NewSource(seed))
	hostile := []float32{
		float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.MaxFloat32), -float32(math.MaxFloat32), 0, 1e30, -1e30,
	}
	abstainedNonFinite := 0
	for c := 0; c < 300; c++ {
		r, a := buildAndLoadProperty(t, rng)
		x := randEmbedding(rng, a.Info.EmbedDim)
		containsNonFinite := false
		for i := range x { // poison a random subset with hostile values
			if rng.Intn(3) == 0 {
				x[i] = hostile[rng.Intn(len(hostile))]
			}
			if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) {
				containsNonFinite = true
			}
		}
		res, err := r.Classify(x) // must not panic
		if err != nil {
			t.Fatalf("seed=%d case=%d: unexpected error: %v", seed, c, err)
		}
		if containsNonFinite {
			// FIXED FINDING 1: the guard must abstain on any non-finite
			// input value — never a non-abstained NaN result.
			if !res.Abstained {
				t.Fatalf("seed=%d case=%d: embedding %v with NaN/±Inf produced non-abstained result %+v", seed, c, x, res)
			}
			if res.Confidence != 0 || res.Margin != 0 || math.IsNaN(res.Confidence) {
				t.Fatalf("seed=%d case=%d: abstained result carries Confidence=%g Margin=%g (want exact 0)",
					seed, c, res.Confidence, res.Margin)
			}
			abstainedNonFinite++
			continue
		}
		if res.Abstained {
			if res.Confidence != 0 || res.Margin != 0 {
				t.Fatalf("seed=%d case=%d: abstained result carries nonzero confidence/margin", seed, c)
			}
			continue
		}
		if math.IsNaN(res.Confidence) || math.IsInf(res.Confidence, 0) {
			t.Fatalf("seed=%d case=%d: finite embedding %v produced non-finite confidence %+v", seed, c, x, res)
		}
		checkShape(t, "20261114", c, res, &a)
	}
	if abstainedNonFinite == 0 {
		t.Fatalf("seed=%d: no case carried a NaN/±Inf embedding; generator cannot reach the guard path", seed)
	}
}

// TestPropertyClassify_Deterministic is P4.
func TestPropertyClassify_Deterministic(t *testing.T) {
	const seed = 20261115
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 300; c++ {
		r, a := buildAndLoadProperty(t, rng)
		x := randEmbedding(rng, a.Info.EmbedDim)
		r1, err1 := r.Classify(x)
		r2, err2 := r.Classify(x)
		if (err1 == nil) != (err2 == nil) {
			t.Fatalf("seed=%d case=%d: divergent errors %v vs %v", seed, c, err1, err2)
		}
		if r1 != r2 {
			t.Fatalf("seed=%d case=%d: Classify not deterministic: %+v vs %+v", seed, c, r1, r2)
		}
	}
}

// TestPropertyClassify_NilReceiver is P6.
func TestPropertyClassify_NilReceiver(t *testing.T) {
	var r *Reservoir
	if _, err := r.Classify([]float32{1}); err == nil {
		t.Fatal("nil receiver must error, not succeed")
	}
}

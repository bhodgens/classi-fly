package reservoir

// Property-based tests for the .fly serialization path — Artifact ->
// WriteFile (fly.go:72-152) -> Load (fly.go:159-317) — hand-rolled with a
// deterministic seeded PRNG and zero new dependencies.
//
// Targets and properties (file:line refs to the implementation contract):
//
//	Target WriteFile/Load round trip over randomized valid artifacts:
//	  P1 round trip: WriteFile followed by Load succeeds for every
//	     validateArtifact-clean artifact and reproduces the metadata
//	     (fly.go:70-71 "never writes a partial file"; Load contract
//	     fly.go:154-158).
//	  P2 quantization bound: |dequant(w) - w| <= scale/2 + 1e-9 for every
//	     weight/readout element that survived the checked int8 quantize
//	     (fly.go:64-68 documented tolerance; quantizeToInt8Checked
//	     fly.go:354-363). Generators deliberately include values at the
//	     ±127*scale boundary so the rejection path is exercised too:
//	     elements the writer REJECTS are skipped (error reported as a
//	     finding), elements it accepts must respect the bound.
//	  P3 load determinism: two Loads of the same written file produce the
//	     same matrixChecksum (fly.go:158 "bit-for-bit identical on every
//	     load"), for both seed- and matrix-mode artifacts.
//	  P4 hostile-artifact no-panic: WriteFile on structurally broken
//	     artifacts (nil/short/long arrays, bad indptr, non-positive or
//	     out-of-range scales, NaN weights, decay outside (0,1)) is always a
//	     clean error, never a panic or a file on disk (fly.go:70-71,
//	     validateArtifact fly.go:366-415).
import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// randArtifact builds a valid randomized artifact, quantization-safe:
// all weights/readout/InW values are within ±maxQ*scale so WriteFile's
// checked quantizer accepts them.
func randArtifact(rng *rand.Rand) Artifact {
	m, _, _ := randCSR(rng)
	n := m.n
	k := 1 + rng.Intn(4)
	d := 1 + rng.Intn(4)

	wScale := 0.01 + rng.Float64()*0.5
	rScale := 0.01 + rng.Float64()*0.5
	// Stay strictly inside the int8 envelope so every element quantizes.
	const qHeadroom = 126
	boundW := qHeadroom * wScale
	boundR := qHeadroom * rScale

	a := Artifact{
		Info: Info{
			Name: "ser", Neurons: n, Edges: len(m.indices), EmbedDim: d,
			Steps: 1 + rng.Intn(6),
			Classes: func() []string {
				cs := []string{"c0", "c1", "c2", "c3"}
				return cs[:k]
			}(),
			License: "CC0-1.0", Attribution: "serialization property test",
		},
		WeightScale:  wScale,
		InScale:      0.1 + rng.Float64(),
		ReadoutScale: rScale,
		Decay:        0.02 + rng.Float64()*0.96,
		Seed:         rng.Uint64(),
		Indptr:       m.indptr,
		Indices:      m.indices,
		Weights:      randWithin(rng, len(m.vals), boundW),
		Readout:      randWithin(rng, n*k, boundR),
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
				th[i] = rng.Float64() * 0.5
			}
			return th
		}(),
	}
	if rng.Intn(2) == 0 {
		a.InW = randWithin(rng, d*n, qHeadroom*a.InScale)
	}
	return a
}

// randWithin returns len values uniformly in [-bound, bound] with boundary
// values seeded in occasionally (ties-to-even corners).
func randWithin(rng *rand.Rand, n int, bound float64) []float64 {
	vs := make([]float64, n)
	for i := range vs {
		switch rng.Intn(8) {
		case 0:
			vs[i] = bound
		case 1:
			vs[i] = -bound
		case 2:
			vs[i] = 0
		default:
			vs[i] = (rng.Float64()*2 - 1) * bound
		}
	}
	return vs
}

// closeEnough reports |got-want| <= scale/2 + eps (the documented
// quantization tolerance, fly.go:66-68).
func closeEnough(got, want, scale float64) bool {
	return math.Abs(got-want) <= scale/2+1e-9
}

// TestPropertySerialize_RoundTripAndQuantization is P1+P2.
func TestPropertySerialize_RoundTripAndQuantization(t *testing.T) {
	const seed = 20261116
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 400; c++ {
		a := randArtifact(rng)
		path := filepath.Join(t.TempDir(), "ser.fly")
		if err := WriteFile(path, a); err != nil {
			t.Fatalf("seed=%d case=%d: WriteFile rejected a valid artifact: %v", seed, c, err)
		}
		r, err := Load(path)
		if err != nil {
			t.Fatalf("seed=%d case=%d: Load of own write failed: %v", seed, c, err)
		}

		// P1: metadata round trip.
		info := r.Info()
		if info.Name != a.Info.Name || info.Neurons != a.Info.Neurons ||
			info.Edges != a.Info.Edges || info.EmbedDim != a.Info.EmbedDim ||
			info.Steps != a.Info.Steps || len(info.Classes) != len(a.Info.Classes) {
			t.Fatalf("seed=%d case=%d: metadata mismatch: %+v", seed, c, info)
		}
		for i := range info.Classes {
			if info.Classes[i] != a.Info.Classes[i] {
				t.Fatalf("seed=%d case=%d: Classes[%d]=%q want %q", seed, c, i, info.Classes[i], a.Info.Classes[i])
			}
		}

		// P2: quantization bound on every dequantized element.
		if len(r.W.vals) != len(a.Weights) {
			t.Fatalf("seed=%d case=%d: edges %d != %d", seed, c, len(r.W.vals), len(a.Weights))
		}
		for i := range a.Weights {
			if !closeEnough(r.W.vals[i], a.Weights[i], a.WeightScale) {
				t.Fatalf("seed=%d case=%d: weights[%d]: dequant %g vs original %g violates |err|<=scale/2 (%g)",
					seed, c, i, r.W.vals[i], a.Weights[i], a.WeightScale)
			}
		}
		if len(r.readout) != len(a.Readout) {
			t.Fatalf("seed=%d case=%d: readout len %d != %d", seed, c, len(r.readout), len(a.Readout))
		}
		for i := range a.Readout {
			if !closeEnough(r.readout[i], a.Readout[i], a.ReadoutScale) {
				t.Fatalf("seed=%d case=%d: readout[%d]: dequant %g vs original %g violates |err|<=scale/2 (%g)",
					seed, c, i, r.readout[i], a.Readout[i], a.ReadoutScale)
			}
		}
		if a.InW != nil {
			if len(r.win) != len(a.InW) {
				t.Fatalf("seed=%d case=%d: win len %d != InW len %d", seed, c, len(r.win), len(a.InW))
			}
			for i := range a.InW {
				if !closeEnough(r.win[i], a.InW[i], a.InScale) {
					t.Fatalf("seed=%d case=%d: InW[%d]: dequant %g vs original %g violates |err|<=scale/2 (%g)",
						seed, c, i, r.win[i], a.InW[i], a.InScale)
				}
			}
		}
	}
}

// TestPropertySerialize_LoadDeterministic is P3.
func TestPropertySerialize_LoadDeterministic(t *testing.T) {
	const seed = 20261117
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 200; c++ {
		a := randArtifact(rng)
		path := filepath.Join(t.TempDir(), "det.fly")
		if err := WriteFile(path, a); err != nil {
			t.Fatalf("seed=%d case=%d: WriteFile: %v", seed, c, err)
		}
		r1, err := Load(path)
		if err != nil {
			t.Fatalf("seed=%d case=%d: Load1: %v", seed, c, err)
		}
		r2, err := Load(path)
		if err != nil {
			t.Fatalf("seed=%d case=%d: Load2: %v", seed, c, err)
		}
		h1, h2 := matrixChecksum(r1), matrixChecksum(r2)
		if h1 != h2 {
			t.Fatalf("seed=%d case=%d: checksum %v != %v across loads", seed, c, h1, h2)
		}
		// A second write of the same artifact must load identically too.
		path2 := filepath.Join(t.TempDir(), "det2.fly")
		if err := WriteFile(path2, a); err != nil {
			t.Fatalf("seed=%d case=%d: WriteFile2: %v", seed, c, err)
		}
		r3, err := Load(path2)
		if err != nil {
			t.Fatalf("seed=%d case=%d: Load3: %v", seed, c, err)
		}
		if h1 != matrixChecksum(r3) {
			t.Fatalf("seed=%d case=%d: checksum differs across writes of the same artifact", seed, c)
		}
	}
}

// TestPropertySerialize_HostileArtifactsNoPanic is P4: WriteFile on
// structurally broken artifacts must return an error and leave no file.
func TestPropertySerialize_HostileArtifactsNoPanic(t *testing.T) {
	const seed = 20261118
	rng := rand.New(rand.NewSource(seed))
	abuse := func(a Artifact, mut func(*Artifact), desc string) error {
		mut(&a)
		path := filepath.Join(t.TempDir(), "hostile.fly")
		err := WriteFile(path, a) // must not panic
		if err == nil {
			if _, statErr := os.Stat(path); statErr == nil {
				t.Fatalf("seed=%d: %s wrote a file at %s", seed, desc, path)
			}
		}
		return err
	}

	for c := 0; c < 150; c++ {
		a := randArtifact(rng)

		// Random truncation/extension of structural arrays. An artifact
		// with zero edges has indptr=[0]; truncation of a 1-element slice
		// to [:0] is itself invalid input to the mutation, so only apply
		// the truncation when there is something to truncate to.
		if err := abuse(a, func(x *Artifact) { x.Indptr = x.Indptr[:len(x.Indptr)/2] }, "truncated indptr"); err == nil {
			t.Fatalf("seed=%d case=%d: truncated indptr accepted", seed, c)
		}
		if len(a.Weights) > 0 {
			if err := abuse(a, func(x *Artifact) { x.Weights = x.Weights[:len(x.Weights)/2] }, "short weights"); err == nil {
				t.Fatalf("seed=%d case=%d: short weights accepted", seed, c)
			}
		}
		if err := abuse(a, func(x *Artifact) { x.Readout = append(x.Readout, 1.0) }, "long readout"); err == nil {
			t.Fatalf("seed=%d case=%d: long readout accepted", seed, c)
		}
		if err := abuse(a, func(x *Artifact) { x.Bias = nil }, "nil bias"); err == nil {
			t.Fatalf("seed=%d case=%d: nil bias accepted", seed, c)
		}
		// indptr[last] != edges (only when indptr is long enough to touch).
		if len(a.Indptr) > 1 {
			if err := abuse(a, func(x *Artifact) {
				x.Indptr[len(x.Indptr)-1] = x.Indptr[len(x.Indptr)-1] + 1
			}, "indptr/edges mismatch"); err == nil {
				t.Fatalf("seed=%d case=%d: indptr/edges mismatch accepted", seed, c)
			}
		}
		if err := abuse(a, func(x *Artifact) { x.WeightScale = 0 }, "zero weight scale"); err == nil {
			t.Fatalf("seed=%d case=%d: zero weight scale accepted", seed, c)
		}
		if err := abuse(a, func(x *Artifact) { x.ReadoutScale = -x.ReadoutScale }, "negative readout scale"); err == nil {
			t.Fatalf("seed=%d case=%d: negative readout scale accepted", seed, c)
		}
		if err := abuse(a, func(x *Artifact) { x.Decay = 1.5 }, "decay 1.5"); err == nil {
			t.Fatalf("seed=%d case=%d: decay 1.5 accepted", seed, c)
		}
		if err := abuse(a, func(x *Artifact) { x.Decay = -0.3 }, "negative decay"); err == nil {
			t.Fatalf("seed=%d case=%d: negative decay accepted", seed, c)
		}
		if err := abuse(a, func(x *Artifact) { x.Info.Steps = 0 }, "zero steps"); err == nil {
			t.Fatalf("seed=%d case=%d: zero steps accepted", seed, c)
		}
		if len(a.Weights) > 0 { // zero-edge artifacts have nothing to poison
			if err := abuse(a, func(x *Artifact) { x.Weights[0] = math.NaN() }, "NaN weight"); err == nil {
				t.Fatalf("seed=%d case=%d: NaN weight accepted", seed, c)
			}
			if err := abuse(a, func(x *Artifact) { x.Weights[0] = math.Inf(1) }, "Inf weight"); err == nil {
				t.Fatalf("seed=%d case=%d: Inf weight accepted", seed, c)
			}
			if err := abuse(a, func(x *Artifact) { x.Weights[0] = 1e9 }, "out-of-range weight"); err == nil {
				t.Fatalf("seed=%d case=%d: out-of-int8-range weight accepted (silent clamp risk)", seed, c)
			}
		}
	}
}

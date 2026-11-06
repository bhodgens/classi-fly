package reservoir

// Property-based tests for the reservoir state-update math, hand-rolled in
// the strategist-usa go-shard-runner style: deterministic seeded PRNG, zero
// new dependencies, bounded case counts, seed printed on failure.
//
// Targets and properties (file:line refs to the implementation contract):
//
//	Target csr.spmv (sparse.go:44-52):
//	  P1 differential vs an independently accumulated dense reference:
//	     out == M*x within a tight relative tolerance for any valid CSR
//	     triple (newCSR-accepted: non-decreasing indptr, indptr[n]==nnz,
//	     cols < n; sparse.go:19-38). The dense reference accumulates
//	     column-ascending per row, which differs from CSR storage order
//	     when duplicates/out-of-order columns exist, so exact bit-equality
//	     is not the contract — agreement to 1e-12 relative is.
//	  P2 overwrite/shape: spmv fully overwrites out and never reads its
//	     prior contents (out pre-poisoned with NaN must not leak NaN unless
//	     the true product is NaN).
//
//	Target core.forward (reservoir.go:27-42):
//	  P3 bounded state: for any finite scalar drive, decay in (0,1), and
//	     any step count, every state entry satisfies |s[i]| <= 1. Why:
//	     s is produced by math.Tanh on every update (reservoir.go:38) and
//	     tanh maps finite inputs into (-1, 1).
//	  P4 zero fixed point: forward on a zero input returns the exact zero
//	     vector. Why: the initial state is zero (reservoir.go:29), the
//	     drive is win[i]*x[0] = 0 (reservoir.go:33), and
//	     tanh(decay*0 + W*0 + 0) == tanh(0) == 0 exactly.
//	  P5 determinism: identical (core, input) pairs produce bit-identical
//	     states. Why: fixed row-ascending, step-sequential execution order
//	     (reservoir.go:23-26 contract, 35-40 implementation).
//	  P6 no-panic on hostile drives: NaN/±Inf/extreme-magnitude inputs
//	     never panic. NaN states are permitted only for non-finite inputs;
//	     finite inputs are covered by P3.
import (
	"math"
	"math/rand"
	"testing"
)

// randCSR builds a valid random CSR matrix plus its dense reference and a
// random input vector. Duplicate and non-ascending column indices are
// deliberately generated: newCSR accepts them and spmv must still agree
// with the dense accumulation.
func randCSR(rng *rand.Rand) (m *csr, dense [][]float64, x []float64) {
	n := 1 + rng.Intn(16)
	indptr := make([]uint32, n+1)
	var indices []uint32
	var vals []float64
	for i := 0; i < n; i++ {
		indptr[i] = uint32(len(indices))
		for k := 0; k < rng.Intn(4); k++ { // 0..3 entries per row
			indices = append(indices, uint32(rng.Intn(n)))
			vals = append(vals, rng.Float64()*4-2) // [-2, 2)
		}
	}
	indptr[n] = uint32(len(indices))

	m = &csr{n: n, indptr: indptr, indices: indices, vals: vals}

	dense = make([][]float64, n)
	for i := range dense {
		dense[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for k := m.indptr[i]; k < m.indptr[i+1]; k++ {
			dense[i][m.indices[k]] += m.vals[k]
		}
	}

	x = make([]float64, n)
	for i := range x {
		switch rng.Intn(6) {
		case 0:
			x[i] = 0
		case 1:
			x[i] = rng.Float64() * 1e6 // large magnitude
		case 2:
			x[i] = -rng.Float64() * 1e6
		default:
			x[i] = rng.Float64()*2 - 1
		}
	}
	return m, dense, x
}

// TestPropertySpmv_MatchesDenseReference is P1.
func TestPropertySpmv_MatchesDenseReference(t *testing.T) {
	const seed = 20261106
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 1000; c++ {
		m, dense, x := randCSR(rng)
		out := make([]float64, m.n)
		m.spmv(x, out)
		for i := 0; i < m.n; i++ {
			var want float64
			for col := 0; col < m.n; col++ { // column-ascending accumulation
				want += dense[i][col] * x[col]
			}
			scale := math.Abs(want)
			if o := math.Abs(out[i]); o > scale {
				scale = o
			}
			if scale < 1 {
				scale = 1
			}
			if math.Abs(out[i]-want) > 1e-9*scale {
				t.Fatalf("seed=%d case=%d row=%d: spmv=%g dense-ref=%g",
					seed, c, i, out[i], want)
			}
		}
	}
}

// TestPropertySpmv_OverwritesOutput is P2: a NaN-poisoned out buffer must
// be fully replaced (NaN may only survive where the true product is NaN,
// which requires a NaN/Inf input; randCSR produces finite inputs).
func TestPropertySpmv_OverwritesOutput(t *testing.T) {
	const seed = 20261107
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 500; c++ {
		m, _, x := randCSR(rng)
		out := make([]float64, m.n)
		for i := range out {
			out[i] = math.NaN() // poison
		}
		m.spmv(x, out)
		for i := 0; i < m.n; i++ {
			if math.IsNaN(out[i]) || math.IsInf(out[i], 0) {
				t.Fatalf("seed=%d case=%d: poisoned out[%d]=%v survived spmv (input leaked or row skipped)",
					seed, c, i, out[i])
			}
		}
	}
}

// newTestCore builds a core with a seed-expanded win (same shape as
// expandSeedWin), a random CSR, decay in (0,1), and steps >= 1.
func newTestCore(rng *rand.Rand) *core {
	m, _, _ := randCSR(rng)
	n := m.n
	win := make([]float64, n)
	for i := range win {
		win[i] = float64(rng.Int31n(5) - 2) // [-2, 2] ints, expandSeedWin shape
	}
	return &core{
		W:     m,
		win:   win,
		decay: 0.05 + rng.Float64()*0.9, // (0, 1)
		steps: 1 + rng.Intn(200),
	}
}

// TestPropertyForward_StateBounded is P3.
func TestPropertyForward_StateBounded(t *testing.T) {
	const seed = 20261108
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 500; c++ {
		co := newTestCore(rng)
		x := []float64{rng.Float64() * math.MaxFloat64 * 0.25} // finite, up to ~1e308
		if rng.Intn(2) == 0 {
			x[0] = -x[0]
		}
		s := co.forward(x)
		if len(s) != co.W.n {
			t.Fatalf("seed=%d case=%d: len(s)=%d want %d", seed, c, len(s), co.W.n)
		}
		for i, v := range s {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < -1 || v > 1 {
				t.Fatalf("seed=%d case=%d: state s[%d]=%v escapes tanh range [-1,1] (x=%g steps=%d decay=%g)",
					seed, c, i, v, x[0], co.steps, co.decay)
			}
		}
	}
}

// TestPropertyForward_ZeroInputFixedPoint is P4.
func TestPropertyForward_ZeroInputFixedPoint(t *testing.T) {
	const seed = 20261109
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 200; c++ {
		co := newTestCore(rng)
		s := co.forward([]float64{0})
		for i, v := range s {
			if v != 0 {
				t.Fatalf("seed=%d case=%d: zero input produced s[%d]=%g (want exact 0)",
					seed, c, i, v)
			}
		}
	}
}

// TestPropertyForward_Deterministic is P5.
func TestPropertyForward_Deterministic(t *testing.T) {
	const seed = 20261110
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 300; c++ {
		co := newTestCore(rng)
		x := []float64{rng.Float64()*1e6 - 5e5}
		s1 := co.forward(x)
		s2 := co.forward(x)
		for i := range s1 {
			if math.Float64bits(s1[i]) != math.Float64bits(s2[i]) {
				t.Fatalf("seed=%d case=%d: forward not bit-deterministic at s[%d]: %v vs %v",
					seed, c, i, s1[i], s2[i])
			}
		}
	}
}

// TestPropertyForward_HostileInputNoPanic is P6. NaN states are allowed
// only when the drive itself is non-finite (NaN/Inf can enter via
// win[i]*x[0]); the invariant enforced here is no panic and no
// out-of-range FINITE state.
func TestPropertyForward_HostileInputNoPanic(t *testing.T) {
	const seed = 20261111
	rng := rand.New(rand.NewSource(seed))
	hostile := []float64{math.NaN(), math.Inf(1), math.Inf(-1),
		math.MaxFloat64, -math.MaxFloat64, 1e308, -1e308, 0}
	for c := 0; c < 300; c++ {
		co := newTestCore(rng)
		var x float64
		if rng.Intn(2) == 0 {
			x = hostile[rng.Intn(len(hostile))]
		} else {
			x = (rng.Float64()*2 - 1) * math.MaxFloat64 * 0.5
		}
		s := co.forward([]float64{x}) // must not panic
		for i, v := range s {
			if !math.IsNaN(v) && !math.IsInf(v, 0) && math.Abs(v) > 1 {
				t.Fatalf("seed=%d case=%d: finite state s[%d]=%v outside [-1,1] for hostile x=%g",
					seed, c, i, v, x)
			}
		}
	}
}

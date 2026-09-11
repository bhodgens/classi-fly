package reservoir

import "math"

// core holds the stateful reservoir recurrence. W is the sparse recurrent
// weight matrix, win a fixed input projection of length n, decay the leak
// term (typically in (0,1)), and steps the number of recurrence iterations
// per forward call.
type core struct {
	W     *csr
	win   []float64 // len n, fixed random projection (seed-derived)
	decay float64
	steps int
}

// forward runs c.steps iterations of the recurrence
//
//	s = tanh(decay*s + W*s + win*x)
//
// over a zero initial state and returns the final state of length n. The
// input x is handled as a scalar drive (win[i]*x[0]) in this leaf; the
// D-dimensional embedding projection lives in the Classify wrapper (a later
// leaf). Execution order is fixed (rows ascending, steps sequential), so the
// output is bit-for-bit deterministic for identical inputs. It allocates its
// own scratch buffers and is safe for concurrent use when distinct cores are
// distinct objects.
func (c *core) forward(x []float64) []float64 {
	n := c.W.n
	s := make([]float64, n) // zero initial state
	recur := make([]float64, n)
	drive := make([]float64, n)
	for i := 0; i < n && i < len(c.win); i++ {
		drive[i] = c.win[i] * x[0] // scalar input in the fixture; see Notes
	}
	for t := 0; t < c.steps; t++ {
		c.W.spmv(s, recur)
		for i := 0; i < n; i++ {
			s[i] = math.Tanh(c.decay*s[i] + recur[i] + drive[i])
		}
	}
	return s
}

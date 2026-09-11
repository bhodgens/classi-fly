package reservoir

import "fmt"

// csr is a compressed-sparse-row matrix with float64 values. It represents a
// square n-by-n matrix (the reservoir recurrence matrix W), stored as the
// standard CSR triple: indptr of length n+1, and column indices plus values of
// length nnz. All three slices must be treated as read-only after construction;
// a *csr is safe for concurrent spmv calls.
type csr struct {
	n       int       // rows (== cols; the reservoir matrix is square)
	indptr  []uint32  // len n+1
	indices []uint32  // len nnz
	vals    []float64 // len nnz
}

// newCSR validates and wraps the CSR triple into a *csr. The input slices are
// retained (not copied), so callers must not mutate them afterwards.
func newCSR(n int, indptr []uint32, indices []uint32, vals []float64) (*csr, error) {
	if n <= 0 {
		return nil, fmt.Errorf("newCSR: n must be positive, got %d", n)
	}
	if len(indptr) != n+1 {
		return nil, fmt.Errorf("newCSR: indptr length %d, want %d", len(indptr), n+1)
	}
	if len(indices) != len(vals) {
		return nil, fmt.Errorf("newCSR: indices length %d != vals length %d", len(indices), len(vals))
	}
	for i := 0; i < n; i++ {
		if indptr[i] > indptr[i+1] {
			return nil, fmt.Errorf("newCSR: indptr[%d]=%d > indptr[%d]=%d (must be non-decreasing)", i, indptr[i], i+1, indptr[i+1])
		}
	}
	if last := indptr[n]; int(last) != len(indices) {
		return nil, fmt.Errorf("newCSR: indptr[%d]=%d does not match nnz=%d", n, last, len(indices))
	}
	return &csr{n: n, indptr: indptr, indices: indices, vals: vals}, nil
}

// spmv computes out = m * x, writing the matrix-vector product into out. Both
// x and out must have length m.n; out is fully overwritten. The inner loop
// accumulates in ascending row order over the CSR layout, so the result is
// bit-for-bit deterministic for identical inputs.
func (m *csr) spmv(x []float64, out []float64) {
	for i := 0; i < m.n; i++ {
		var sum float64
		for k := m.indptr[i]; k < m.indptr[i+1]; k++ {
			sum += m.vals[k] * x[m.indices[k]]
		}
		out[i] = sum
	}
}

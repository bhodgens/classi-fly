package reservoir

import (
	"hash/fnv"
	"math"
	"os"
)

// The helpers below exist only for the fly tests in this package.

// osReadAll reads a whole file (thin os.ReadFile wrapper for readability at
// the call sites).
func osReadAll(path string) ([]byte, error) { return os.ReadFile(path) }

// osWriteFile writes a whole file (thin os.WriteFile wrapper for readability
// at the call sites).
func osWriteFile(path string, b []byte) error { return os.WriteFile(path, b, 0o644) }

// matrixChecksum returns a position-sensitive hash over every float64 that
// defines a loaded reservoir's numeric behavior: W (indptr, indices, vals),
// win, readout, bias, threshold, plus decay and readoutScale. Two loads of
// the same file must produce identical checksums; any drift in layout or
// dequantization changes the hash.
func matrixChecksum(r *Reservoir) uint64 {
	h := fnv.New64a()
	absorb := func(vs ...float64) {
		var buf [8]byte
		for _, v := range vs {
			bits := math.Float64bits(v)
			for i := 0; i < 8; i++ {
				buf[i] = byte(bits >> (8 * i))
			}
			h.Write(buf[:])
		}
	}
	absorb(float64(r.W.n))
	for _, v := range r.W.indptr {
		absorb(float64(v))
	}
	for _, v := range r.W.indices {
		absorb(float64(v))
	}
	absorb(r.W.vals...)
	absorb(r.win...)
	absorb(r.readout...)
	absorb(r.bias...)
	absorb(r.threshold...)
	absorb(r.decay, r.readoutScale)
	return h.Sum64()
}

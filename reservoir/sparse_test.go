package reservoir

import "testing"

func TestCSR_Spmv(t *testing.T) {
	// 3x3 matrix:
	//  2 0 1
	//  0 3 0
	//  4 0 5
	m, err := newCSR(3,
		[]uint32{0, 2, 3, 5},
		[]uint32{0, 2, 1, 0, 2},
		[]float64{2, 1, 3, 4, 5},
	)
	if err != nil {
		t.Fatalf("newCSR: %v", err)
	}
	out := make([]float64, 3)
	m.spmv([]float64{1, 1, 1}, out)
	want := []float64{3, 3, 9}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], want[i])
		}
	}
}

func TestCSR_RejectsBadShape(t *testing.T) {
	if _, err := newCSR(3, []uint32{0, 2}, []uint32{0}, []float64{1}); err == nil {
		t.Fatal("expected error for indptr length mismatch")
	}
}

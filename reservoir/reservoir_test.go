package reservoir

import "testing"

func TestCore_ForwardShapes(t *testing.T) {
	m, _ := newCSR(2, []uint32{0, 1, 2}, []uint32{1, 0}, []float64{1, 1})
	c := &core{W: m, win: []float64{1, 0}, decay: 0.9, steps: 4}
	got := c.forward([]float64{0.5})
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	for _, v := range got {
		if v != v || v > 1 || v < -1 {
			t.Fatalf("state out of range: %v", v)
		}
	}
}

func TestCore_ForwardDeterministic(t *testing.T) {
	m, _ := newCSR(3, []uint32{0, 1, 2, 3}, []uint32{1, 2, 0}, []float64{0.5, 0.5, 0.5})
	c := &core{W: m, win: []float64{1, 0.5, 0.25}, decay: 0.8, steps: 6}
	a := c.forward([]float64{0.3})
	b := c.forward([]float64{0.3})
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestCore_ForwardStableOver100Calls(t *testing.T) {
	m, err := newCSR(4,
		[]uint32{0, 2, 3, 5, 6},
		[]uint32{1, 3, 2, 0, 2, 1},
		[]float64{0.3, 0.1, -0.4, 0.2, 0.5, -0.2},
	)
	if err != nil {
		t.Fatalf("newCSR: %v", err)
	}
	c := &core{W: m, win: []float64{1, 0.5, 0.25, 0.125}, decay: 0.85, steps: 8}
	ref := c.forward([]float64{0.7})
	for call := 0; call < 100; call++ {
		got := c.forward([]float64{0.7})
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("call %d: nondeterministic at %d: %v vs %v", call, i, got[i], ref[i])
			}
		}
	}
}

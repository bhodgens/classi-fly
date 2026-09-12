package reservoir

import (
	"os"
	"runtime"
	"testing"
)

// productionArtifact is a real embed_dim=1024 / 2952-neuron artifact built from
// the larval connectome (data/prod_model.fly). Benchmarks skip when absent so
// the suite stays runnable without the 34 MB connectome download.
const productionArtifact = "../data/prod_model.fly"

func loadProduction(b *testing.B) *Reservoir {
	b.Helper()
	if _, err := os.Stat(productionArtifact); err != nil {
		b.Skip("production artifact not present")
	}
	r, err := Load(productionArtifact)
	if err != nil {
		b.Fatalf("Load: %v", err)
	}
	return r
}

func BenchmarkLoadProduction(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r, err := Load(productionArtifact)
		if err != nil {
			b.Fatalf("Load: %v", err)
		}
		_ = r
	}
}

func BenchmarkClassifyProduction(b *testing.B) {
	r := loadProduction(b)
	emb := make([]float32, r.Info().EmbedDim)
	for i := range emb {
		emb[i] = 0.03125 // unit-norm-ish constant; value is irrelevant to timing
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := r.Classify(emb); err != nil {
			b.Fatalf("Classify: %v", err)
		}
	}
}

// TestFootprintReport prints the resident heap cost of holding one loaded
// artifact. Run with: go test ./reservoir/ -run TestFootprintReport -v
func TestFootprintReport(t *testing.T) {
	if _, err := os.Stat(productionArtifact); err != nil {
		t.Skip("production artifact not present")
	}
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	r, err := Load(productionArtifact)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	info := r.Info()

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	delta := after.HeapAlloc - before.HeapAlloc
	t.Logf("artifact %s: neurons=%d edges=%d embed_dim=%d classes=%d",
		productionArtifact, info.Neurons, info.Edges, info.EmbedDim, len(info.Classes))
	t.Logf("heap delta after load: %.1f MB (%d bytes)", float64(delta)/1e6, delta)
	st, err := os.Stat(productionArtifact)
	if err == nil {
		t.Logf("on-disk artifact: %.2f MB", float64(st.Size())/1e6)
	}
	keepAlive = r // prevent collection before the measurement is read
}

var keepAlive *Reservoir

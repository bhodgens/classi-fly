package reservoir

import (
	"fmt"
	"math"
)

// InputMode enumerates the two ways a .fly artifact can encode its fixed
// input projection (Contract 1 payload "input_mode" byte).
type InputMode uint8

const (
	// InputModeSeed derives win from math/rand.NewSource(seed): win[i] is
	// drawn as one uniform int in [-2, 2] per neuron from that source,
	// expanded by the producer's in_scale. Used when the generator owns the
	// projection and the artifact should stay small.
	InputModeSeed InputMode = 0
	// InputModeMatrix stores the projection explicitly as int8[D*N]
	// (embedding-major, see WriteFile) plus in_scale.
	InputModeMatrix InputMode = 1
)

// Result is one classification outcome (Contract 2). When Abstained is true
// the caller should fall through to another strategy; Confidence and Margin
// are then zero, but Class still names the winning lane for diagnostics.
type Result struct {
	Class      string
	Confidence float64 // winning lane probability, 0..1
	Margin     float64 // top1 - top2 probability
	Abstained  bool    // true when the caller should fall through
}

// Info is the artifact metadata surfaced by Reservoir.Info (Contract 2).
type Info struct {
	Name        string
	Neurons     int
	Edges       int
	EmbedDim    int
	Steps       int
	Classes     []string
	License     string
	Attribution string
}

// Reservoir is a loaded .fly classifier. All state is unexported and
// immutable after Load, so a single *Reservoir is safe for concurrent
// Classify calls.
type Reservoir struct {
	info Info

	W            *csr
	decay        float64
	steps        int
	win          []float64 // length neurons; fixed input projection
	readout      []float64 // length neurons*classes, row-major neuron-major
	readoutScale float64
	bias         []float64 // length classes
	threshold    []float64 // length classes
	classes      []string
}

// Classify runs the full pipeline on one D-dimensional embedding:
// project (u[i] = sum_j in_w[j*N+i] * x[j]) -> reservoir recurrence
// (s = tanh(decay*s + W*s + u), zero initial state, steps iterations) ->
// readout logits (l[c] = dot(readout[:, c], s) * readout_scale + bias[c]) ->
// softmax -> top-1/top-2 with margin. It abstains (Abstained=true, zero
// Confidence/Margin) when the top-1 probability is below the winning
// class's threshold. Dimension mismatch is an error. Deterministic: fixed
// iteration order, no map iteration, no shared mutable state.
func (r *Reservoir) Classify(embedding []float32) (Result, error) {
	if r == nil {
		return Result{}, fmt.Errorf("reservoir: Classify on nil Reservoir")
	}
	if len(embedding) != r.info.EmbedDim {
		return Result{}, fmt.Errorf("reservoir: embedding dim %d != artifact embed_dim %d",
			len(embedding), r.info.EmbedDim)
	}
	n := r.info.Neurons
	k := len(r.classes)

	// Input projection. Matrix mode stores embedding-major in_w[j*N+i]
	// (length D*N): u[i] = sum_j in_w[j*N+i] * x[j]. Seed mode stores the
	// expanded projection neuron-major (length N): u[i] = win[i] * x[0].
	u := make([]float64, n)
	if len(r.win) == n*len(embedding) {
		for j := 0; j < len(embedding); j++ {
			xj := float64(embedding[j])
			if xj == 0 {
				continue
			}
			base := j * n
			for i := 0; i < n; i++ {
				u[i] += r.win[base+i] * xj
			}
		}
	} else {
		x0 := float64(embedding[0])
		for i := 0; i < n && i < len(r.win); i++ {
			u[i] = r.win[i] * x0
		}
	}

	// Reservoir recurrence (same order of operations as core.forward).
	s := make([]float64, n)
	recur := make([]float64, n)
	for t := 0; t < r.steps; t++ {
		r.W.spmv(s, recur)
		for i := 0; i < n; i++ {
			s[i] = math.Tanh(r.decay*s[i] + recur[i] + u[i])
		}
	}

	// Readout logits, then a numerically stable softmax.
	logits := make([]float64, k)
	for i := 0; i < n; i++ {
		si := s[i]
		if si == 0 {
			continue
		}
		base := i * k
		for c := 0; c < k; c++ {
			logits[c] += r.readout[base+c] * si
		}
	}
	maxLogit := math.Inf(-1)
	for c := 0; c < k; c++ {
		logits[c] = logits[c]*r.readoutScale + r.bias[c]
		if logits[c] > maxLogit {
			maxLogit = logits[c]
		}
	}
	sum := 0.0
	probs := make([]float64, k)
	for c := 0; c < k; c++ {
		p := math.Exp(logits[c] - maxLogit)
		probs[c] = p
		sum += p
	}
	for c := 0; c < k; c++ {
		probs[c] /= sum
	}

	// Top-1 and top-2 by scan (deterministic tie-break: lowest class index
	// wins; floats are compared with the shared epsilon, never != in a
	// sort comparator).
	const eps = 1e-9
	top1, top2 := 0, -1
	for c := 1; c < k; c++ {
		d := probs[c] - probs[top1]
		if d > eps {
			top2 = top1
			top1 = c
		} else if top2 == -1 || probs[c] > probs[top2]+eps {
			top2 = c
		}
	}
	confidence := probs[top1]
	margin := 1 - confidence
	if top2 >= 0 {
		margin = confidence - probs[top2]
	}

	if confidence < r.threshold[top1] {
		return Result{Class: r.classes[top1], Confidence: 0, Margin: 0, Abstained: true}, nil
	}
	return Result{Class: r.classes[top1], Confidence: confidence, Margin: margin}, nil
}

// Info returns a copy of the artifact metadata. Mutating the returned
// Classes slice does not affect the receiver.
func (r *Reservoir) Info() Info {
	if r == nil {
		return Info{}
	}
	out := r.info
	out.Classes = append([]string(nil), r.info.Classes...)
	return out
}

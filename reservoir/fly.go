package reservoir

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/caimlas/classi-fly/internal/flyformat"
)

// defaultDecay is the leak term used when an artifact header carries no
// decay extension. It matches the Python tooling's DEFAULT_DECAY so both
// runtimes classify identically on legacy files.
const defaultDecay = 0.8

// Artifact is the producer-side description of a .fly file: everything
// WriteFile needs to emit a complete Contract 1 container. Weight arrays
// are float64; WriteFile quantizes them to int8 using the matching scale
// (quantized to nearest, ties to even; per-element error is then at most
// half that scale).
type Artifact struct {
	Info Info

	WeightScale  float64   // int8 -> float64 scale for CSR weights
	InScale      float64   // int8 -> float64 scale for in_w (matrix mode)
	ReadoutScale float64   // int8 -> float64 scale for the readout matrix
	Decay        float64   // leak term; (0,1); 0 writes no header extension
	Seed         uint64    // input_mode=seed: seed for the win expansion
	InputMode    InputMode // forced mode when InW is nil

	Indptr  []uint32  // length Neurons+1
	Indices []uint32  // length Edges
	Weights []float64 // length Edges, pre-quantization

	InW []float64 // matrix mode: length EmbedDim*Neurons, embedding-major (see WriteFile)

	Readout   []float64 // length Neurons*len(Classes), neuron-major
	Bias      []float64 // length len(Classes)
	Threshold []float64 // per-class route floor, length len(Classes)

	Source     string
	CreatedUTC string
}

// WriteFile encodes artifact into the versioned .fly container at path
// (Contract 1: FLYRES01 magic, little-endian uint32 header length, JSON
// header, zstd-compressed payload).
//
// Payload order is frozen: indptr (u32, N+1), indices (u32, E), weights
// (i8, E), input_mode (u8), then either the seed (u64) or the input matrix
// (i8[D*N] + in_row_len u32 + in_scale f64), then the readout (i8, N*K),
// readout_scale (f64), bias (f32, K), threshold (f32, K).
//
// Input projection layout: in matrix mode the in_w array is stored
// EMBEDDING-MAJOR as in_w[j*N + i] — D consecutive neuron blocks, where
// element j*N+i scales embedding[j] into neuron i (win[i] =
// sum_j in_w[j*N+i] * x[j]). in_row_len records the stride (N). This
// matches the Python trainer's expansion and keeps the dot product in
// Classify cache-friendly. Seed mode stores just Seed; Load expands it via
// math/rand.NewSource.
//
// Quantization: each float is divided by its scale and rounded to the
// nearest int8 with ties to even, then dequantized by multiplying the
// scale back. The documented tolerance is therefore |dequant(w) - w| <=
// scale/2 for every element (|w/scale| <= 127 required; values outside the
// int8 range are an error, never a silent clamp).
//
// All structural inconsistencies (length mismatches, out-of-range weights,
// degenerate dimensions) are errors; WriteFile never writes a partial file.
func WriteFile(path string, a Artifact) error {
	if err := validateArtifact(&a); err != nil {
		return err
	}
	h := flyformat.Header{
		Format:      flyformat.FormatString,
		Version:     flyformat.FormatVersion,
		Name:        a.Info.Name,
		Neurons:     a.Info.Neurons,
		Edges:       a.Info.Edges,
		EmbedDim:    a.Info.EmbedDim,
		Steps:       a.Info.Steps,
		Classes:     a.Info.Classes,
		WeightScale: a.WeightScale,
		Source:      a.Source,
		License:     a.Info.License,
		Attribution: a.Info.Attribution,
		CreatedUTC:  a.CreatedUTC,
	}
	if a.Decay != 0 {
		h.Decay = a.Decay
	}
	headerJSON, err := flyformat.MarshalHeader(h)
	if err != nil {
		return err
	}

	n, d := a.Info.Neurons, a.Info.EmbedDim
	k := len(a.Info.Classes)

	payload := make([]byte, 0, payloadCap(n, d, k, int(a.Info.Edges)))
	for _, v := range a.Indptr {
		payload = binary.LittleEndian.AppendUint32(payload, v)
	}
	for _, v := range a.Indices {
		payload = binary.LittleEndian.AppendUint32(payload, v)
	}
	for i, v := range a.Weights {
		q, err := quantizeToInt8Checked(v / a.WeightScale)
		if err != nil {
			return fmt.Errorf("reservoir: weights[%d]: %w", i, err)
		}
		payload = append(payload, byte(q))
	}
	payload = append(payload, byte(a.inputMode()))
	if a.inputMode() == InputModeMatrix {
		for j := 0; j < d; j++ {
			for i := 0; i < n; i++ {
				v := a.InW[j*n+i]
				q, err := quantizeToInt8Checked(v / a.InScale)
				if err != nil {
					return fmt.Errorf("reservoir: in_w[%d]: %w", j*n+i, err)
				}
				payload = append(payload, byte(q))
			}
		}
		payload = binary.LittleEndian.AppendUint32(payload, uint32(n))
		payload = binary.LittleEndian.AppendUint64(payload, math.Float64bits(a.InScale))
	} else {
		payload = binary.LittleEndian.AppendUint64(payload, a.Seed)
	}
	for i := 0; i < n; i++ {
		for c := 0; c < k; c++ {
			v := a.Readout[i*k+c]
			q, err := quantizeToInt8Checked(v / a.ReadoutScale)
			if err != nil {
				return fmt.Errorf("reservoir: readout[%d]: %w", i*k+c, err)
			}
			payload = append(payload, byte(q))
		}
	}
	payload = binary.LittleEndian.AppendUint64(payload, math.Float64bits(a.ReadoutScale))
	for _, v := range a.Bias {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(float32(v)))
	}
	for _, v := range a.Threshold {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(float32(v)))
	}

	return flyformat.WriteFile(path, headerJSON, payload)
}

// Load reads and fully validates a .fly artifact and returns a ready
// Reservoir, or an error. It is fail-closed: a missing, corrupt, truncated,
// or internally inconsistent file produces an error and no reservoir, never
// a zero value that would silently misclassify. The in-memory matrix built
// from a given file is bit-for-bit identical on every load.
func Load(path string) (*Reservoir, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reservoir: load %s: %w", path, err)
	}
	headerJSON, payload, err := flyformat.DecodeContainer(blob)
	if err != nil {
		return nil, fmt.Errorf("reservoir: load %s: %w", path, err)
	}
	h, err := flyformat.UnmarshalHeader(headerJSON)
	if err != nil {
		return nil, fmt.Errorf("reservoir: load %s: %w", path, err)
	}

	n, e, d := h.Neurons, h.Edges, h.EmbedDim
	k := len(h.Classes)

	need := func(count int, what string) error {
		if len(payload) < count {
			return fmt.Errorf("reservoir: load %s: payload truncated in %s (need %d, have %d)",
				path, what, count, len(payload))
		}
		return nil
	}

	minLen := 4*(n+1) + 4*e + e + 1
	if err := need(minLen, "arrays"); err != nil {
		return nil, err
	}

	indptr := make([]uint32, n+1)
	for i := range indptr {
		indptr[i] = binary.LittleEndian.Uint32(payload[4*i:])
	}
	indices := make([]uint32, e)
	for i := range indices {
		indices[i] = binary.LittleEndian.Uint32(payload[4*(n+1)+4*i:])
	}
	wRaw := payload[4*(n+1)+4*e : 4*(n+1)+4*e+e]
	vals := make([]float64, e)
	for i, b := range wRaw {
		vals[i] = float64(int8(b)) * h.WeightScale
	}
	pos := 4*(n+1) + 4*e + e

	inputMode := InputMode(payload[pos])
	pos++
	var win []float64
	switch inputMode {
	case InputModeSeed:
		if err := need(pos+8, "seed"); err != nil {
			return nil, err
		}
		seed := binary.LittleEndian.Uint64(payload[pos:])
		pos += 8
		win = expandSeedWin(seed, n)
	case InputModeMatrix:
		if err := need(pos+d*n+4+8, "input matrix"); err != nil {
			return nil, err
		}
		inRaw := payload[pos : pos+d*n]
		pos += d * n
		rowLen := binary.LittleEndian.Uint32(payload[pos:])
		pos += 4
		if int(rowLen) != n {
			return nil, fmt.Errorf("reservoir: load %s: in_row_len %d != neurons %d", path, rowLen, n)
		}
		inScale := math.Float64frombits(binary.LittleEndian.Uint64(payload[pos:]))
		pos += 8
		if inScale <= 0 || math.IsNaN(inScale) {
			return nil, fmt.Errorf("reservoir: load %s: bad in_scale %g", path, inScale)
		}
		// in_w is stored embedding-major: in_w[j*N + i] drives neuron i
		// from embedding[j]. Load expands it to the neuron-major win vector
		// Classify consumes.
		win = make([]float64, d*n)
		for j := 0; j < d; j++ {
			xbase := j * n
			for i := 0; i < n; i++ {
				win[xbase+i] = float64(int8(inRaw[xbase+i])) * inScale
			}
		}
	default:
		return nil, fmt.Errorf("reservoir: load %s: unknown input_mode %d", path, inputMode)
	}

	if err := need(pos+n*k+8+8*k, "readout"); err != nil {
		return nil, err
	}
	// Contract 1 payload order: the readout int8 matrix comes first, its
	// f64 scale follows.
	readout := make([]float64, n*k)
	for i, b := range payload[pos : pos+n*k] {
		readout[i] = float64(int8(b))
	}
	pos += n * k
	readoutScale := math.Float64frombits(binary.LittleEndian.Uint64(payload[pos:]))
	pos += 8
	if readoutScale <= 0 || math.IsNaN(readoutScale) {
		return nil, fmt.Errorf("reservoir: load %s: bad readout_scale %g", path, readoutScale)
	}
	for i := range readout {
		readout[i] *= readoutScale
	}

	bias := make([]float64, k)
	for c := 0; c < k; c++ {
		bias[c] = float64(math.Float32frombits(binary.LittleEndian.Uint32(payload[pos:])))
		pos += 4
	}
	threshold := make([]float64, k)
	for c := 0; c < k; c++ {
		threshold[c] = float64(math.Float32frombits(binary.LittleEndian.Uint32(payload[pos:])))
		pos += 4
	}
	if pos != len(payload) {
		return nil, fmt.Errorf("reservoir: load %s: %d trailing payload bytes (corrupt artifact)",
			path, len(payload)-pos)
	}

	decay := h.Decay
	if decay == 0 {
		decay = defaultDecay
	}

	m, err := newCSR(n, indptr, indices, vals)
	if err != nil {
		return nil, fmt.Errorf("reservoir: load %s: %w", path, err)
	}
	for _, idx := range indices {
		if int(idx) >= n {
			return nil, fmt.Errorf("reservoir: load %s: column index %d out of range (neurons=%d)",
				path, idx, n)
		}
	}

	r := &Reservoir{
		info: Info{
			Name:        h.Name,
			Neurons:     n,
			Edges:       e,
			EmbedDim:    d,
			Steps:       h.Steps,
			Classes:     append([]string(nil), h.Classes...),
			License:     h.License,
			Attribution: h.Attribution,
		},
		W:            m,
		decay:        decay,
		steps:        h.Steps,
		win:          win,
		readout:      readout,
		readoutScale: readoutScale,
		bias:         bias,
		threshold:    threshold,
		classes:      append([]string(nil), h.Classes...),
	}
	return r, nil
}

// inputMode reports the effective input mode: matrix whenever an explicit
// projection is present, otherwise the stored/seed mode.
func (a *Artifact) inputMode() InputMode {
	if a.InW != nil {
		return InputModeMatrix
	}
	if a.InputMode != 0 {
		return a.InputMode
	}
	return InputModeSeed
}

// expandSeedWin derives the neuron-major win vector from seed exactly the
// way the Python trainer expands seed-mode fixtures (math/rand.Int31n-
// compatible uniform ints in [-2, 2], scaled by the artifact's in_scale).
// The stream depends only on the seed, so every load produces the same
// projection.
func expandSeedWin(seed uint64, n int) []float64 {
	rng := rand.New(rand.NewSource(int64(seed)))
	win := make([]float64, n)
	for i := range win {
		win[i] = float64(rng.Int31n(5) - 2)
	}
	return win
}

// quantizeToInt8 rounds v/scale to the nearest int8 with ties to even. It
// assumes scale > 0 and |v| small enough for int8; the checked wrapper
// below guards the range.
func quantizeToInt8(x float64) int8 {
	return int8(math.RoundToEven(x))
}

// quantizeToInt8Checked rejects values that would not survive the int8
// round trip (range overflow, NaN, infinites).
func quantizeToInt8Checked(x float64) (int8, error) {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0, fmt.Errorf("value %g is not quantizable to int8", x)
	}
	r := math.RoundToEven(x)
	if r < -128 || r > 127 {
		return 0, fmt.Errorf("value %g exceeds int8 range after scaling", x)
	}
	return int8(r), nil
}

// validateArtifact checks every structural invariant WriteFile depends on.
func validateArtifact(a *Artifact) error {
	n, e, d := a.Info.Neurons, a.Info.Edges, a.Info.EmbedDim
	k := len(a.Info.Classes)
	if n <= 0 || e < 0 || d <= 0 {
		return fmt.Errorf("reservoir: degenerate artifact dims (neurons=%d edges=%d embed_dim=%d)", n, e, d)
	}
	if k == 0 {
		return fmt.Errorf("reservoir: artifact has no classes")
	}
	if a.Info.Steps <= 0 {
		return fmt.Errorf("reservoir: steps must be positive, got %d", a.Info.Steps)
	}
	if len(a.Indptr) != n+1 {
		return fmt.Errorf("reservoir: indptr length %d, want %d", len(a.Indptr), n+1)
	}
	if len(a.Indices) != e || len(a.Weights) != e {
		return fmt.Errorf("reservoir: indices/weights length mismatch (edges=%d)", e)
	}
	if last := a.Indptr[n]; int(last) != e {
		return fmt.Errorf("reservoir: indptr[N]=%d != edges %d", last, e)
	}
	if a.WeightScale <= 0 || a.ReadoutScale <= 0 {
		return fmt.Errorf("reservoir: weight/readout scales must be positive")
	}
	if len(a.Bias) != k || len(a.Threshold) != k {
		return fmt.Errorf("reservoir: bias/threshold length must match classes (%d)", k)
	}
	if len(a.Readout) != n*k {
		return fmt.Errorf("reservoir: readout length %d, want %d", len(a.Readout), n*k)
	}
	switch a.inputMode() {
	case InputModeMatrix:
		if len(a.InW) != d*n {
			return fmt.Errorf("reservoir: in_w length %d, want embed_dim*neurons = %d", len(a.InW), d*n)
		}
		if a.InScale <= 0 {
			return fmt.Errorf("reservoir: in_scale must be positive in matrix mode")
		}
	case InputModeSeed:
		if a.InW != nil {
			return fmt.Errorf("reservoir: seed mode cannot carry in_w")
		}
	default:
		return fmt.Errorf("reservoir: unknown input mode %d", a.InputMode)
	}
	if a.Decay != 0 && (a.Decay <= 0 || a.Decay >= 1) {
		return fmt.Errorf("reservoir: decay must be in (0, 1), got %g", a.Decay)
	}
	return nil
}

// payloadCap estimates the payload buffer size to avoid repeated growth.
func payloadCap(n, d, k, e int) int {
	return 4*(n+1) + 4*e + e + 1 + 8 + d*n + 4 + 8 + n*k + 8 + 8*k
}

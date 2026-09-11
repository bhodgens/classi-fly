package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"time"

	"github.com/caimlas/classi-fly/reservoir"
)

// errNonCommercial marks an adjacency whose license forbids commercial use;
// runPack maps it to the license exit code.
var errNonCommercial = errors.New("non-commercial artifact refused; re-run with --allow-noncommercial to override")

// flyMagic is the .fly container magic (Contract 1).
const flyMagic = "FLYRES01"

// defaultDecay is the leak term written into the .fly header when --decay is
// not given. It must match the offline trainer's default so the runtime
// recurrence and the training-time recurrence agree.
const defaultDecay = 0.8

// adjacencyExport is the Contract 3 adjacency shape emitted by tools/ingest.
type adjacencyExport struct {
	Name        string    `json:"name"`
	Neurons     int       `json:"neurons"`
	Edges       int       `json:"edges"`
	Indptr      []uint32  `json:"indptr"`
	Indices     []uint32  `json:"indices"`
	Weights     []float64 `json:"weights"`
	Source      string    `json:"source"`
	License     string    `json:"license"`
	Attribution string    `json:"attribution"`
}

// readoutExport is the Contract 3 readout shape emitted by tools/train.
type readoutExport struct {
	Classes      []string    `json:"classes"`
	W            [][]float64 `json:"W"`
	Bias         []float64   `json:"bias"`
	Threshold    []float64   `json:"threshold"`
	WeightScale  float64     `json:"weight_scale"`
	ReadoutScale float64     `json:"readout_scale"`
}

// packBlock carries what the Contract 3 exports cannot: the classifier's
// input contract (embed_dim, steps) and the input projection, which the .fly
// payload stores per Contract 1's input_mode (matrix in_w + in_scale, or a
// seed the runtime expands). See README.md for the JSON shape.
type packBlock struct {
	EmbedDim  int     `json:"embed_dim"`
	Steps     int     `json:"steps"`
	InputMode string  `json:"input_mode"` // "seed" or "matrix"
	Seed      uint64  `json:"seed,omitempty"`
	InW       []int8  `json:"in_w,omitempty"`
	InScale   float64 `json:"in_scale,omitempty"`
}

// validate checks the pack block against the adjacency's neuron count.
func (p packBlock) validate(n int) error {
	if p.EmbedDim <= 0 {
		return fmt.Errorf("pack block: embed_dim must be positive, got %d", p.EmbedDim)
	}
	if p.Steps <= 0 {
		return fmt.Errorf("pack block: steps must be positive, got %d", p.Steps)
	}
	switch p.InputMode {
	case "matrix":
		if len(p.InW) != p.EmbedDim*n {
			return fmt.Errorf("pack block: in_w length %d, want embed_dim*neurons = %d", len(p.InW), p.EmbedDim*n)
		}
		if p.InScale <= 0 {
			return fmt.Errorf("pack block: in_scale must be positive for matrix mode")
		}
	case "seed":
		// Any uint64 seed is valid; the runtime expands it.
	default:
		return fmt.Errorf(`pack block: input_mode must be "seed" or "matrix", got %q`, p.InputMode)
	}
	return nil
}

// runPack implements `classi-fly pack`.
func runPack(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("pack", flag.ContinueOnError)
	fs.SetOutput(stderr)
	adjPath := fs.String("adjacency", "", "adjacency export JSON (Contract 3)")
	roPath := fs.String("readout", "", "readout export JSON (Contract 3)")
	packPath := fs.String("pack", "", "pack-block JSON (embed_dim, steps, input projection; see README)")
	outPath := fs.String("out", "", "output .fly path")
	decay := fs.Float64("decay", defaultDecay, "leak term written to the artifact header (must match training)")
	allowNC := fs.Bool("allow-noncommercial", false, "permit adjacency sources licensed CC-BY-NC-4.0")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	if *adjPath == "" || *roPath == "" || *packPath == "" || *outPath == "" {
		fmt.Fprintln(stderr, "classi-fly pack: --adjacency, --readout, --pack and --out are required")
		return exitUsage
	}
	if *decay <= 0 || *decay >= 1 {
		fmt.Fprintf(stderr, "classi-fly pack: --decay must be in (0, 1), got %v\n", *decay)
		return exitUsage
	}
	var adj adjacencyExport
	if err := readJSONFile(*adjPath, &adj); err != nil {
		fmt.Fprintf(stderr, "classi-fly pack: reading adjacency export: %v\n", err)
		return exitLoad
	}
	var ro readoutExport
	if err := readJSONFile(*roPath, &ro); err != nil {
		fmt.Fprintf(stderr, "classi-fly pack: reading readout export: %v\n", err)
		return exitLoad
	}
	var pb packBlock
	if err := readJSONFile(*packPath, &pb); err != nil {
		fmt.Fprintf(stderr, "classi-fly pack: reading pack block: %v\n", err)
		return exitLoad
	}
	if err := packRun(adj, ro, pb, *decay, *allowNC, *outPath); err != nil {
		if errors.Is(err, errNonCommercial) {
			fmt.Fprintf(stderr, "classi-fly pack: %v\n", err)
			return exitLicense
		}
		fmt.Fprintf(stderr, "classi-fly pack: %v\n", err)
		return exitLoad
	}
	if st, err := os.Stat(*outPath); err == nil {
		fmt.Fprintf(stdout, "wrote %s (%d bytes)\n", *outPath, st.Size())
	} else {
		fmt.Fprintf(stdout, "wrote %s\n", *outPath)
	}
	return exitOK
}

// packRun validates the two exports plus the pack block, applies the license
// gate, and writes the .fly via reservoir.WriteFile. Graph weights, the
// input projection and the readout are passed through pre-quantization:
// WriteFile performs the int8 quantization (v/scale, clamped to int8), so
// each element lands within one quantization step of its input.
func packRun(adj adjacencyExport, ro readoutExport, pb packBlock, decay float64, allowNonCommercial bool, outPath string) error {
	if adj.Edges == 0 {
		adj.Edges = len(adj.Indices)
	}
	if err := validateExports(adj, ro); err != nil {
		return err
	}
	if !allowNonCommercial && adj.License == "CC-BY-NC-4.0" {
		return errNonCommercial
	}
	if err := pb.validate(adj.Neurons); err != nil {
		return err
	}
	classes := append([]string(nil), ro.Classes...)

	// int8-friendly scales, trainer convention: max|v|/127 floored, so the
	// largest magnitude fills the int8 range. The readout export already
	// carries its scale (tools/train writes max|W|/127); use it verbatim.
	adjScale := maxAbs(adj.Weights) / 127
	if adjScale < 1e-6 {
		adjScale = 1e-6
	}
	roScale := ro.ReadoutScale
	if roScale <= 0 {
		roScale = 1
	}

	flat := make([]float64, 0, adj.Neurons*len(classes))
	for _, row := range ro.W {
		flat = append(flat, row...)
	}

	mode := reservoir.InputModeSeed
	if pb.InputMode == "matrix" {
		mode = reservoir.InputModeMatrix
	}
	var inW []float64
	if mode == reservoir.InputModeMatrix {
		// The pack block stores the int8-encoded projection plus its scale;
		// Artifact.InW carries the dequantized projection (WriteFile
		// requantizes as v/InScale, so the round trip is byte-exact).
		inW = make([]float64, len(pb.InW))
		for i, v := range pb.InW {
			inW[i] = float64(v) * pb.InScale
		}
	}

	art := reservoir.Artifact{
		Info: reservoir.Info{
			Name:        adj.Name,
			Neurons:     adj.Neurons,
			Edges:       adj.Edges,
			EmbedDim:    pb.EmbedDim,
			Steps:       pb.Steps,
			Classes:     classes,
			License:     adj.License,
			Attribution: adj.Attribution,
		},
		WeightScale:  adjScale,
		InScale:      pb.InScale,
		ReadoutScale: roScale,
		Decay:        decay,
		Seed:         pb.Seed,
		InputMode:    mode,
		Indptr:       adj.Indptr,
		Indices:      adj.Indices,
		Weights:      adj.Weights,
		InW:          inW,
		Readout:      flat,
		Bias:         append([]float64(nil), ro.Bias...),
		Threshold:    append([]float64(nil), ro.Threshold...),
		Source:       adj.Source,
		CreatedUTC:   time.Now().UTC().Format(time.RFC3339),
	}
	return reservoir.WriteFile(outPath, art)
}

// validateExports enforces the dimension agreements the CLI contract pins:
// the readout class list labels the artifact, the readout has one row per
// neuron, and every row, bias and threshold entry counts one per class.
func validateExports(adj adjacencyExport, ro readoutExport) error {
	if adj.Neurons <= 0 {
		return fmt.Errorf("adjacency: neurons must be positive, got %d", adj.Neurons)
	}
	if len(adj.Indptr) != adj.Neurons+1 {
		return fmt.Errorf("adjacency: indptr length %d, want neurons+1 = %d", len(adj.Indptr), adj.Neurons+1)
	}
	if len(adj.Indices) != len(adj.Weights) {
		return fmt.Errorf("adjacency: indices length %d != weights length %d", len(adj.Indices), len(adj.Weights))
	}
	if adj.Edges != len(adj.Indices) {
		return fmt.Errorf("adjacency: edges %d != len(indices) %d", adj.Edges, len(adj.Indices))
	}
	if len(ro.Classes) == 0 {
		return fmt.Errorf("readout: classes must not be empty")
	}
	if len(ro.W) != adj.Neurons {
		return fmt.Errorf("readout: %d W rows, want one per adjacency neuron (%d)", len(ro.W), adj.Neurons)
	}
	for i, row := range ro.W {
		if len(row) != len(ro.Classes) {
			return fmt.Errorf("readout: W row %d has %d entries, want one per class (%d)", i, len(row), len(ro.Classes))
		}
	}
	if len(ro.Bias) != len(ro.Classes) {
		return fmt.Errorf("readout: bias length %d, want one per class (%d)", len(ro.Bias), len(ro.Classes))
	}
	if len(ro.Threshold) != len(ro.Classes) {
		return fmt.Errorf("readout: threshold length %d, want one per class (%d)", len(ro.Threshold), len(ro.Classes))
	}
	return nil
}

// maxAbs returns the largest absolute value in v (0 for an empty slice).
func maxAbs(v []float64) float64 {
	var m float64
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

// readJSONFile reads and decodes a JSON document from path.
func readJSONFile(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// reorderFlagSet moves positional arguments after all flags so the contract
// syntax `classify model.fly --embedding ...` parses with stdlib flag, which
// otherwise stops at the first positional. Value-taking flags keep their
// next argument attached (detected from the FlagSet: any flag whose Value is
// not a stdlib bool flag consumes a value); "--" terminates flag parsing.
func reorderFlagSet(fs *flag.FlagSet, args []string) []string {
	takesValue := map[string]bool{}
	fs.VisitAll(func(f *flag.Flag) {
		if _, isBool := f.Value.(interface{ IsBoolFlag() bool }); !isBool {
			takesValue[f.Name] = true
		}
	})
	var flags, positional []string
	terminated := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case terminated:
			positional = append(positional, a)
			continue
		case a == "--":
			terminated = true
			continue
		case len(a) > 1 && a[0] == '-':
			flags = append(flags, a)
			name := a[1:]
			if len(name) > 1 && name[0] == '-' {
				name = name[1:]
			}
			if eq := strings.IndexByte(name, '='); eq >= 0 {
				continue // value attached with =
			}
			if takesValue[name] && i+1 < len(args) {
				i++
				flags = append(flags, args[i])
			}
		default:
			positional = append(positional, a)
		}
	}
	out := append([]string(nil), flags...)
	if terminated {
		out = append(out, "--")
	}
	return append(out, positional...)
}

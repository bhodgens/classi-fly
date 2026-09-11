package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/caimlas/classi-fly/reservoir"
)

// classifyOutput is the JSON shape `classify` prints (frozen key set).
type classifyOutput struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	Margin     float64 `json:"margin"`
	Abstained  bool    `json:"abstained"`
}

// runClassify implements `classi-fly classify`.
func runClassify(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	emb := fs.String("embedding", "", "comma-separated embedding values, e.g. 0.1,0.2,0.3")
	embFile := fs.String("embedding-file", "", "JSON file holding the embedding as an array of numbers")
	allowNC := fs.Bool("allow-noncommercial", false, "permit non-commercial artifacts")
	if err := fs.Parse(reorderFlagSet(fs, args)); err != nil {
		return exitUsage
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "classi-fly classify: exactly one .fly path is required")
		return exitUsage
	}
	if (*emb == "") == (*embFile == "") {
		fmt.Fprintln(stderr, "classi-fly classify: pass exactly one of --embedding or --embedding-file")
		return exitUsage
	}
	var values []float32
	var err error
	if *emb != "" {
		values, err = parseEmbedding(*emb)
	} else {
		values, err = readEmbeddingFile(*embFile)
	}
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: %v\n", err)
		return exitUsage
	}

	_, header, err := readFlyHeader(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: %v\n", err)
		return exitLoad
	}
	if !*allowNC && isNonCommercial(header.License) {
		fmt.Fprintln(stderr, "classi-fly classify: non-commercial artifact refused; re-run with --allow-noncommercial to override")
		return exitLicense
	}

	r, err := reservoir.Load(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: %v\n", err)
		return exitLoad
	}

	res, err := r.Classify(values)
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: %v\n", err)
		return exitLoad
	}
	out := classifyOutput{
		Class:      res.Class,
		Confidence: res.Confidence,
		Margin:     res.Margin,
		Abstained:  res.Abstained,
	}
	b, err := json.Marshal(out)
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: encoding result: %v\n", err)
		return exitLoad
	}
	if _, err := fmt.Fprintln(stdout, string(b)); err != nil {
		fmt.Fprintf(stderr, "classi-fly classify: writing output: %v\n", err)
		return exitLoad
	}
	return exitOK
}

// parseEmbedding parses a comma-separated float32 vector.
func parseEmbedding(s string) ([]float32, error) {
	parts := strings.Split(s, ",")
	out := make([]float32, 0, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return nil, fmt.Errorf("--embedding: empty value at position %d", i+1)
		}
		v, err := strconv.ParseFloat(p, 32)
		if err != nil {
			return nil, fmt.Errorf("--embedding: %q is not a number", p)
		}
		out = append(out, float32(v))
	}
	return out, nil
}

// readEmbeddingFile reads a JSON array of numbers as the embedding.
func readEmbeddingFile(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw []float64
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("%s: want a JSON array of numbers: %w", path, err)
	}
	out := make([]float32, len(raw))
	for i, v := range raw {
		out[i] = float32(v)
	}
	return out, nil
}

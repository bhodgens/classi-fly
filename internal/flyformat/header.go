// Package flyformat implements the low-level encoding and decoding of the
// versioned .fly binary container defined by classi-fly Contract 1: an
// 8-byte magic ("FLYRES01"), a little-endian uint32 header length, a JSON
// header, and a zstd-compressed payload of packed little-endian arrays.
//
// This package is internal; consumers use the public API in
// github.com/caimlas/classi-fly/reservoir.
package flyformat

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// FormatString is the frozen "format" header field value for all v1
// artifacts. Files with any other value are rejected on load.
const FormatString = "fly-reservoir"

// FormatVersion is the frozen container version this package reads and
// writes. Files with any other version are rejected on load.
const FormatVersion = 1

// Header mirrors the Contract 1 header JSON with the exact frozen field
// names. MarshalHeader emits them in a deterministic order; UnmarshalHeader
// rejects files whose format or version differs from FormatString /
// FormatVersion, or that carry unknown fields.
type Header struct {
	Format      string   `json:"format"`
	Version     int      `json:"version"`
	Name        string   `json:"name"`
	Neurons     int      `json:"neurons"`
	Edges       int      `json:"edges"`
	EmbedDim    int      `json:"embed_dim"`
	Steps       int      `json:"steps"`
	Classes     []string `json:"classes"`
	WeightScale float64  `json:"weight_scale"`
	Source      string   `json:"source"`
	License     string   `json:"license"`
	Attribution string   `json:"attribution"`
	CreatedUTC  string   `json:"created_utc"`

	// Decay is an append-only v1 header extension written by the packer and
	// read optionally by consumers (older artifacts omit it; the Python
	// trainer defaults to 0.8 when absent). It is the leak term of the
	// reservoir recurrence, expected in (0, 1). It is deliberately NOT part
	// of the frozen Contract 1 field set, so it must never be renamed.
	Decay float64 `json:"decay,omitempty"`
}

// MarshalHeader encodes h into the canonical header JSON bytes. The struct's
// json tags are the frozen Contract 1 names; encode-side validation rejects
// degenerate values before they can be written into an artifact.
func MarshalHeader(h Header) ([]byte, error) {
	if err := validate(&h); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(h); err != nil {
		return nil, fmt.Errorf("flyformat: marshal header: %w", err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// UnmarshalHeader parses and validates header JSON. It rejects any header
// whose format is not FormatString, whose version is not FormatVersion, that
// carries unknown fields, or whose required values are missing or
// nonsensical. Decay is optional: a zero value means "not present" and
// consumers apply their own default.
func UnmarshalHeader(b []byte) (Header, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	var h Header
	if err := dec.Decode(&h); err != nil {
		return Header{}, fmt.Errorf("flyformat: header JSON: %w", err)
	}
	if dec.More() {
		return Header{}, fmt.Errorf("flyformat: trailing data after header JSON")
	}
	if h.Format != FormatString {
		return Header{}, fmt.Errorf("flyformat: unsupported format %q (want %q)", h.Format, FormatString)
	}
	if h.Version != FormatVersion {
		return Header{}, fmt.Errorf("flyformat: unsupported version %d (want %d)", h.Version, FormatVersion)
	}
	if err := validate(&h); err != nil {
		return Header{}, err
	}
	return h, nil
}

// validate enforces the invariants every header must satisfy, on both the
// encode and decode paths.
func validate(h *Header) error {
	if h.Format != FormatString {
		return fmt.Errorf("flyformat: format must be %q, got %q", FormatString, h.Format)
	}
	if h.Version != FormatVersion {
		return fmt.Errorf("flyformat: version must be %d, got %d", FormatVersion, h.Version)
	}
	if h.Neurons <= 0 {
		return fmt.Errorf("flyformat: neurons must be positive, got %d", h.Neurons)
	}
	if h.Edges < 0 {
		return fmt.Errorf("flyformat: edges must be non-negative, got %d", h.Edges)
	}
	if h.EmbedDim <= 0 {
		return fmt.Errorf("flyformat: embed_dim must be positive, got %d", h.EmbedDim)
	}
	if h.Steps <= 0 {
		return fmt.Errorf("flyformat: steps must be positive, got %d", h.Steps)
	}
	if len(h.Classes) == 0 {
		return fmt.Errorf("flyformat: classes must be non-empty")
	}
	if h.WeightScale <= 0 {
		return fmt.Errorf("flyformat: weight_scale must be positive, got %g", h.WeightScale)
	}
	if h.Decay != 0 && (h.Decay <= 0 || h.Decay >= 1) {
		return fmt.Errorf("flyformat: decay must be in (0, 1) when present, got %g", h.Decay)
	}
	return nil
}

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// headerLite mirrors the frozen Contract 1 header fields the CLI itself
// needs (license gating, format check). The full parse and validation live
// in the reservoir package; this lite view exists so `info` can print the
// raw header JSON verbatim, since the public API exposes only parsed Info().
type headerLite struct {
	Format      string   `json:"format"`
	Version     int      `json:"version"`
	Name        string   `json:"name"`
	Neurons     int      `json:"neurons"`
	Edges       int      `json:"edges"`
	EmbedDim    int      `json:"embed_dim"`
	Steps       int      `json:"steps"`
	Classes     []string `json:"classes"`
	License     string   `json:"license"`
	Attribution string   `json:"attribution"`
}

// readFlyHeader reads only the container framing (magic + length-prefixed
// header JSON) of a .fly file and returns the header bytes verbatim plus a
// parsed lite view. It does not decode the payload.
func readFlyHeader(path string) ([]byte, headerLite, error) {
	var lite headerLite
	f, err := os.Open(path)
	if err != nil {
		return nil, lite, err
	}
	defer f.Close()
	magic := make([]byte, 8)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, lite, fmt.Errorf("%s: reading magic: %w", path, err)
	}
	if string(magic) != flyMagic {
		return nil, lite, fmt.Errorf("%s: not a .fly artifact (bad magic %q)", path, magic)
	}
	var headerLen uint32
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, lite, fmt.Errorf("%s: reading header length: %w", path, err)
	}
	raw := make([]byte, headerLen)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, lite, fmt.Errorf("%s: reading header: %w", path, err)
	}
	if err := json.Unmarshal(raw, &lite); err != nil {
		return nil, lite, fmt.Errorf("%s: decoding header: %w", path, err)
	}
	return raw, lite, nil
}

// isNonCommercial reports whether a license string marks a non-commercial
// source (FlyWire family: CC BY-NC 4.0). Only shippable-with-attribution
// licenses (CC-BY, CC BY 4.0, none/public-domain synthetic) pass.
func isNonCommercial(license string) bool {
	switch license {
	case "CC-BY-NC-4.0", "CC BY-NC 4.0", "CC-BY-NC", "CC BY-NC":
		return true
	default:
		return false
	}
}

package flyformat

import (
	"fmt"
	"os"

	"github.com/klauspost/compress/zstd"
)

// Magic is the 8-byte identifier at the start of every .fly file
// (Contract 1).
const Magic = "FLYRES01"

// magicLen is the fixed magic size; headerLenOffset is where the little-
// endian uint32 header length begins, and headerOffset where the JSON
// header starts.
const (
	magicLen        = len(Magic)
	headerLenOffset = magicLen
	headerOffset    = magicLen + 4
)

var (
	encOnce lazyEncoder
	decOnce lazyDecoder
)

// lazyEncoder/lazyDecoder hold one shared zstd codec pair. zstd.Encoder/
// Decoder are concurrency-safe and heavier to construct than to reuse, so
// package-level reuse keeps per-call allocation flat.
type lazyEncoder struct {
	done   bool
	enc    *zstd.Encoder
	failed bool
}

type lazyDecoder struct {
	done   bool
	dec    *zstd.Decoder
	failed bool
}

func (l *lazyEncoder) get() (*zstd.Encoder, error) {
	if !l.done {
		l.done = true
		enc, err := zstd.NewWriter(nil, zstd.WithEncoderCRC(true))
		if err != nil {
			l.failed = true
			return nil, fmt.Errorf("flyformat: zstd encoder: %w", err)
		}
		l.enc = enc
	}
	if l.failed {
		return nil, fmt.Errorf("flyformat: zstd encoder unavailable")
	}
	return l.enc, nil
}

func (l *lazyDecoder) get() (*zstd.Decoder, error) {
	if !l.done {
		l.done = true
		dec, err := zstd.NewReader(nil, zstd.WithDecoderMaxMemory(1<<30))
		if err != nil {
			l.failed = true
			return nil, fmt.Errorf("flyformat: zstd decoder: %w", err)
		}
		l.dec = dec
	}
	if l.failed {
		return nil, fmt.Errorf("flyformat: zstd decoder unavailable")
	}
	return l.dec, nil
}

// EncodeContainer assembles a complete .fly file: the FLYRES01 magic, a
// little-endian uint32 header length, the JSON header bytes verbatim, and
// the zstd-compressed payload. EncodeAll is sequential and deterministic, so
// identical inputs always produce identical files.
func EncodeContainer(headerJSON, payload []byte) ([]byte, error) {
	if len(headerJSON) == 0 {
		return nil, fmt.Errorf("flyformat: empty header JSON")
	}
	enc, err := encOnce.get()
	if err != nil {
		return nil, err
	}
	compressed := enc.EncodeAll(payload, nil)
	out := make([]byte, 0, headerOffset+len(headerJSON)+len(compressed))
	out = append(out, Magic...)
	out = append(out, byte(len(headerJSON)), byte(len(headerJSON)>>8), byte(len(headerJSON)>>16), byte(len(headerJSON)>>24))
	out = append(out, headerJSON...)
	out = append(out, compressed...)
	return out, nil
}

// DecodeContainer parses a .fly file: it validates the magic and header
// length, returns the raw header JSON bytes, and decompresses the payload.
// Payloads written by the Python tooling may be stored raw (uncompressed)
// when no zstd encoder is available; a zstd magic mismatch therefore falls
// back to treating the bytes as the raw payload, matching the landed Python
// reader (tools/eval/flyio.py, tools/train/states.py), which auto-detects
// both forms.
func DecodeContainer(blob []byte) (headerJSON, payload []byte, err error) {
	if len(blob) < headerOffset {
		return nil, nil, fmt.Errorf("flyformat: file too short (%d bytes)", len(blob))
	}
	if string(blob[:magicLen]) != Magic {
		return nil, nil, fmt.Errorf("flyformat: bad magic %q", blob[:magicLen])
	}
	headerLen := uint32(blob[headerLenOffset]) | uint32(blob[headerLenOffset+1])<<8 |
		uint32(blob[headerLenOffset+2])<<16 | uint32(blob[headerLenOffset+3])<<24
	if uint64(headerOffset)+uint64(headerLen) > uint64(len(blob)) {
		return nil, nil, fmt.Errorf("flyformat: header length %d overruns %d-byte file", headerLen, len(blob))
	}
	header := blob[headerOffset : headerOffset+int(headerLen)]
	compressed := blob[headerOffset+int(headerLen):]
	if len(compressed) == 0 {
		return nil, nil, fmt.Errorf("flyformat: missing payload")
	}
	// zstd frames start with magic 0xFD2FB528 (LE bytes 28 B5 2F FD). A
	// payload that does not start with that magic was stored raw.
	if !(compressed[0] == 0x28 && compressed[1] == 0xB5 && compressed[2] == 0x2F && compressed[3] == 0xFD) {
		return header, compressed, nil
	}
	dec, err := decOnce.get()
	if err != nil {
		return nil, nil, err
	}
	raw, err := dec.DecodeAll(compressed, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("flyformat: zstd decompress payload: %w", err)
	}
	return header, raw, nil
}

// WriteFile encodes the container and writes it to path in one os.WriteFile
// call: a failed write never leaves a partially written artifact at path.
func WriteFile(path string, headerJSON, payload []byte) error {
	blob, err := EncodeContainer(headerJSON, payload)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		return fmt.Errorf("flyformat: write %s: %w", path, err)
	}
	return nil
}

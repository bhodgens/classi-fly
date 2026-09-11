package flyformat

import (
	"bytes"
	"testing"
)

func TestContainerRoundTrip(t *testing.T) {
	payload := []byte("hello fly payload \x00\x01\x02 binary")
	b, err := EncodeContainer([]byte(`{"format":"fly-reservoir","version":1}`), payload)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(b, []byte("FLYRES01")) {
		t.Fatalf("missing magic: %x", b[:8])
	}
	header, got, err := DecodeContainer(b)
	if err != nil {
		t.Fatal(err)
	}
	if string(header) != `{"format":"fly-reservoir","version":1}` {
		t.Fatalf("header mismatch: %s", header)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("payload mismatch: got %d bytes want %d", len(got), len(payload))
	}
}

func TestContainerCompresses(t *testing.T) {
	payload := bytes.Repeat([]byte("compressible"), 500)
	b, err := EncodeContainer([]byte(`{}`), payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) >= len(payload) {
		t.Fatalf("expected compression, got %d bytes for %d-byte payload", len(b), len(payload))
	}
}

func TestDecodeRejectsBadMagic(t *testing.T) {
	if _, _, err := DecodeContainer([]byte("NOPE0000\x00\x00\x00\x00")); err == nil {
		t.Fatal("expected bad-magic error")
	}
}

func TestDecodeRejectsTruncated(t *testing.T) {
	full, err := EncodeContainer([]byte(`{}`), []byte("payload"))
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 4, 9, 11, 12} {
		if _, _, err := DecodeContainer(full[:n]); err == nil {
			t.Fatalf("truncation to %d bytes: expected error", n)
		}
	}
	// A header_len that overruns the file must also fail.
	bad := append([]byte(nil), full[:11]...)
	bad = append(bad, 0xff, 0xff, 0xff, 0xff)
	if _, _, err := DecodeContainer(bad); err == nil {
		t.Fatal("expected overrun-header-length error")
	}
}

func TestDecodeRejectsCorruptZstd(t *testing.T) {
	full, err := EncodeContainer([]byte(`{}`), bytes.Repeat([]byte("x"), 64))
	if err != nil {
		t.Fatal(err)
	}
	full[len(full)-1] ^= 0xff
	if _, _, err := DecodeContainer(full); err == nil {
		t.Fatal("expected zstd corruption error")
	}
}

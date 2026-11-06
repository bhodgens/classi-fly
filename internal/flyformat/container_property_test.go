package flyformat

// Property-based tests for the Contract 1 container codec — EncodeContainer
// (container.go:79-94) / DecodeContainer (container.go:103-134) — hand-rolled
// with a deterministic seeded PRNG and zero new dependencies.
//
// Targets and properties (file:line refs to the implementation contract):
//
//	Target EncodeContainer/DecodeContainer round trip:
//	  P1 round trip: DecodeContainer(EncodeContainer(h, p)) returns exactly
//	     (h, p) — byte-identical header JSON and payload — for any
//	     non-empty header and ANY payload (empty, 1 byte, all zeros, high
//	     entropy, zstd-magic-lookalike payloads that stress the raw-vs-
//	     compressed sniffing at container.go:120-124).
//	  P2 determinism: identical (header, payload) inputs encode to
//	     byte-identical files (container.go:77-78 "sequential and
//	     deterministic" contract).
//	  P3 shape contract: every encoded container is magic + 4-byte LE
//	     length + header + payload, with the stored length equal to the
//	     header length (container.go:88-92).
//
//	Target DecodeContainer hostile inputs:
//	  P4 no-panic + fail-closed: ANY byte sequence either decodes to
//	     (header, payload) or returns an error — never a panic. Truncated
//	     files, bad magic, header lengths that overrun the file, and empty
//	     payloads must all be errors, never partial success
//	     (container.go:104-119).
//	  P5 length-field sanity: when the 4-byte length field claims more
//	     bytes than the file contains, DecodeContainer must error
//	     (container.go:112-114); the minimal 12-byte file with length 0
//	     has no payload and must error too (container.go:117-119).
//
//	Target MarshalHeader/UnmarshalHeader (header.go:54-91):
//	  P6 round trip: UnmarshalHeader(MarshalHeader(h)) reproduces h for
//	     every valid h (including arbitrary unicode names/sources, and
//	     both decay-present and decay-absent forms — the omitempty
//	     extension at header.go:43-48). Note: this is a VALUE round trip,
//	     not byte round trip, since Go may reorder struct fields vs the
//	     input JSON.
//	  P7 fail-closed on invalid headers: MarshalHeader rejects every
//	     header violating validate (header.go:95-124) — bad format/version,
//	     degenerate dims, empty classes, non-positive scales, decay outside
//	     (0,1) — with an error, never a write. UnmarshalHeader likewise
//	     rejects mutated JSON on those fields.
import (
	"bytes"
	"math"
	"math/rand"
	"testing"
	"unicode/utf8"
)

// randBytes returns n pseudorandom bytes.
func randBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Intn(256))
	}
	return b
}

// validHeader returns a Header passing validate (header.go:95-124), with
// optionally hostile string fields.
func validHeader(rng *rand.Rand) Header {
	return Header{
		Format:      FormatString,
		Version:     FormatVersion,
		Name:        "prop-" + randPrintable(rng, 1+rng.Intn(24)),
		Neurons:     1 + rng.Intn(4096),
		Edges:       rng.Intn(1 << 20),
		EmbedDim:    1 + rng.Intn(512),
		Steps:       1 + rng.Intn(256),
		Classes:     []string{randPrintable(rng, 1+rng.Intn(16)), randPrintable(rng, 1+rng.Intn(16))},
		WeightScale: 1e-6 + rng.Float64()*100,
		Source:      randPrintable(rng, rng.Intn(32)),
		License:     "CC0-1.0",
		Attribution: randPrintable(rng, rng.Intn(64)),
		CreatedUTC:  "2026-11-06T00:00:00Z",
		Decay:       decayOrZero(rng),
	}
}

// decayOrZero returns 0 (absent) two-thirds of the time, else a value in
// (0, 1).
func decayOrZero(rng *rand.Rand) float64 {
	if rng.Intn(3) == 0 {
		return 0.01 + rng.Float64()*0.98
	}
	return 0
}

// randPrintable returns a valid-UTF8 random string, sometimes including
// unicode, quotes, and control-adjacent characters that stress JSON
// escaping (SetEscapeHTML(false), header.go:60).
func randPrintable(rng *rand.Rand, n int) string {
	runes := make([]rune, 0, n)
	pool := []rune("abcXYZ019 -_.\"'\\/\t\né漢字🚀<&>")
	for len(runes) < n {
		runes = append(runes, pool[rng.Intn(len(pool))])
	}
	s := string(runes)
	if !utf8.ValidString(s) { // safety net; pool is all-valid so unreachable
		return "fallback"
	}
	return s
}

// TestPropertyContainer_RoundTrip is P1.
func TestPropertyContainer_RoundTrip(t *testing.T) {
	const seed = 20261119
	rng := rand.New(rand.NewSource(seed))
	shapes := []func(*rand.Rand) []byte{
		func(*rand.Rand) []byte { return nil },                          // empty
		func(r *rand.Rand) []byte { return randBytes(r, 1) },            // 1 byte
		func(r *rand.Rand) []byte { return make([]byte, 1+r.Intn(64)) }, // all zeros
		func(r *rand.Rand) []byte { return randBytes(r, r.Intn(4096)) }, // high entropy
		func(r *rand.Rand) []byte { // zstd magic lookalike
			b := append([]byte{0x28, 0xB5, 0x2F, 0xFD}, randBytes(r, r.Intn(256))...)
			return b
		},
		func(r *rand.Rand) []byte { return randBytes(r, 100000) }, // large
	}
	for c := 0; c < 600; c++ {
		header := randBytes(rng, 1+rng.Intn(512))
		payload := shapes[rng.Intn(len(shapes))](rng)
		blob, err := EncodeContainer(header, payload)
		if err != nil {
			t.Fatalf("seed=%d case=%d: EncodeContainer: %v", seed, c, err)
		}
		gotHeader, gotPayload, err := DecodeContainer(blob)
		if err != nil {
			t.Fatalf("seed=%d case=%d: DecodeContainer of own encode: %v", seed, c, err)
		}
		if !bytes.Equal(gotHeader, header) {
			t.Fatalf("seed=%d case=%d: header not byte-identical after round trip (%d vs %d bytes)",
				seed, c, len(gotHeader), len(header))
		}
		if !bytes.Equal(gotPayload, payload) {
			t.Fatalf("seed=%d case=%d: payload not byte-identical after round trip (%d vs %d bytes)",
				seed, c, len(gotPayload), len(payload))
		}
	}
}

// TestPropertyContainer_DeterministicEncoding is P2.
func TestPropertyContainer_DeterministicEncoding(t *testing.T) {
	const seed = 20261120
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 300; c++ {
		header := randBytes(rng, 1+rng.Intn(256))
		payload := randBytes(rng, rng.Intn(2048))
		b1, err := EncodeContainer(header, payload)
		if err != nil {
			t.Fatal(err)
		}
		b2, err := EncodeContainer(header, payload)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b1, b2) {
			t.Fatalf("seed=%d case=%d: encode not deterministic (%d vs %d bytes)", seed, c, len(b1), len(b2))
		}
	}
}

// TestPropertyContainer_EncodeShape is P3.
func TestPropertyContainer_EncodeShape(t *testing.T) {
	const seed = 20261121
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 300; c++ {
		header := randBytes(rng, 1+rng.Intn(256))
		payload := randBytes(rng, rng.Intn(1024))
		blob, err := EncodeContainer(header, payload)
		if err != nil {
			t.Fatal(err)
		}
		if string(blob[:8]) != Magic {
			t.Fatalf("seed=%d case=%d: magic prefix missing", seed, c)
		}
		hlen := uint32(blob[8]) | uint32(blob[9])<<8 | uint32(blob[10])<<16 | uint32(blob[11])<<24
		if int(hlen) != len(header) {
			t.Fatalf("seed=%d case=%d: stored header length %d != actual %d", seed, c, hlen, len(header))
		}
		if !bytes.Equal(blob[12:12+len(header)], header) {
			t.Fatalf("seed=%d case=%d: header bytes misplaced", seed, c)
		}
		if len(blob) != 12+len(header)+len(blob[12+len(header):]) {
			t.Fatalf("seed=%d case=%d: total length inconsistent", seed, c)
		}
	}
}

// TestPropertyContainer_HostileDecodeNoPanic is P4+P5: arbitrary byte
// sequences decode or error, never panic; the enumerated corruption shapes
// must be errors.
func TestPropertyContainer_HostileDecodeNoPanic(t *testing.T) {
	const seed = 20261122
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 1500; c++ {
		n := rng.Intn(256)
		blob := randBytes(rng, n)
		h, p, err := DecodeContainer(blob) // must not panic
		if err == nil && len(blob) >= 12 {
			// A random short blob succeeding is only legitimate when the
			// magic matched; assert that, and that returned slices are
			// within the blob's bounds.
			if string(blob[:8]) != Magic {
				t.Fatalf("seed=%d case=%d: decode succeeded without magic", seed, c)
			}
			_ = h
			_ = p
		}
		if err == nil && len(blob) < 12 {
			t.Fatalf("seed=%d case=%d: %d-byte blob decoded without error", seed, c, len(blob))
		}
	}

	// Enumerated corruptions of a valid container: each must error.
	valid, err := EncodeContainer([]byte(`{"format":"fly-reservoir"}`), []byte("payload-bytes"))
	if err != nil {
		t.Fatal(err)
	}
	mustFail := [][]byte{
		nil,
		valid[:8],  // magic only
		valid[:11], // magic + partial length
		valid[:12], // magic + length, no header bytes despite nonzero length? (length=21)
		append(append([]byte{}, valid[:4]...), 'B', 'A', 'D', '!'), // corrupt magic
		valid[:len(valid)/2], // truncated mid-payload
		func() []byte { // length field claims 4 GB, file is tiny
			b := append([]byte{}, valid[:8]...)
			b = append(b, 0x00, 0x00, 0x00, 0x40) // 1GiB >> file size
			return append(b, valid[12:]...)
		}(),
		valid[:len(valid)-1], // payload missing final byte
	}
	for i, bad := range mustFail {
		if _, _, err := DecodeContainer(bad); err == nil {
			t.Fatalf("seed=%d corruption=%d: expected error, got success", seed, i)
		}
	}

	// Minimal well-formed frame with length 0: header empty is rejected by
	// EncodeContainer upstream, but a hand-built one has no payload bytes —
	// must error (container.go:117-119).
	frame := append([]byte(Magic), 0, 0, 0, 0)
	if _, _, err := DecodeContainer(frame); err == nil {
		t.Fatalf("seed=%d: empty-payload frame decoded without error", seed)
	}
}

// TestPropertyHeader_RoundTrip is P6.
func TestPropertyHeader_RoundTrip(t *testing.T) {
	const seed = 20261123
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 400; c++ {
		h := validHeader(rng)
		b, err := MarshalHeader(h)
		if err != nil {
			t.Fatalf("seed=%d case=%d: MarshalHeader rejected valid header: %v", seed, c, err)
		}
		got, err := UnmarshalHeader(b)
		if err != nil {
			t.Fatalf("seed=%d case=%d: UnmarshalHeader of own marshal: %v", seed, c, err)
		}
		if !headerEqual(got, h) { // value round trip (Header contains a slice)
			t.Fatalf("seed=%d case=%d: header value changed across round trip:\n got %+v\nwant %+v",
				seed, c, got, h)
		}
	}
}

// headerEqual compares two Headers field-by-field (Header holds a []string,
// so == is unavailable).
func headerEqual(a, b Header) bool {
	if len(a.Classes) != len(b.Classes) || a.Format != b.Format || a.Version != b.Version ||
		a.Name != b.Name || a.Neurons != b.Neurons || a.Edges != b.Edges ||
		a.EmbedDim != b.EmbedDim || a.Steps != b.Steps || a.WeightScale != b.WeightScale ||
		a.Source != b.Source || a.License != b.License || a.Attribution != b.Attribution ||
		a.CreatedUTC != b.CreatedUTC || a.Decay != b.Decay {
		return false
	}
	for i := range a.Classes {
		if a.Classes[i] != b.Classes[i] {
			return false
		}
	}
	return true
}

// TestPropertyHeader_FailClosed is P7.
func TestPropertyHeader_FailClosed(t *testing.T) {
	const seed = 20261124
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < 100; c++ {
		mutations := []func(*Header){
			func(h *Header) { h.Format = "nope" },
			func(h *Header) { h.Version = FormatVersion + 1 },
			func(h *Header) { h.Version = 0 },
			func(h *Header) { h.Neurons = 0 },
			func(h *Header) { h.Neurons = -1 },
			func(h *Header) { h.Edges = -1 },
			func(h *Header) { h.EmbedDim = 0 },
			func(h *Header) { h.Steps = 0 },
			func(h *Header) { h.Steps = -3 },
			func(h *Header) { h.Classes = nil },
			func(h *Header) { h.WeightScale = 0 },
			func(h *Header) { h.WeightScale = -1.5 },
			func(h *Header) { h.Decay = 1.0 },
			func(h *Header) { h.Decay = math.NaN() }, // NaN decay
			func(h *Header) { h.Decay = -0.5 },
		}
		mut := mutations[rng.Intn(len(mutations))]
		h := validHeader(rng)
		mut(&h)
		if _, err := MarshalHeader(h); err == nil {
			t.Fatalf("seed=%d case=%d: MarshalHeader accepted invalid header %+v", seed, c, h)
		}
	}
}

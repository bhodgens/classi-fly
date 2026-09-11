package flyformat

import (
	"bytes"
	"testing"
)

func TestHeaderRoundTrip(t *testing.T) {
	h := Header{Format: "fly-reservoir", Version: 1, Name: "t",
		Neurons: 8, Edges: 10, EmbedDim: 4, Steps: 3,
		Classes: []string{"a", "b"}, WeightScale: 0.01,
		Source: "synthetic", License: "none", Attribution: "n/a",
		CreatedUTC: "2026-09-11T00:00:00Z"}
	b, err := MarshalHeader(h)
	if err != nil {
		t.Fatal(err)
	}
	got, err := UnmarshalHeader(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Neurons != 8 || got.WeightScale != 0.01 || len(got.Classes) != 2 {
		t.Fatalf("mismatch: %+v", got)
	}
	if got.Name != "t" || got.Source != "synthetic" || got.License != "none" ||
		got.Attribution != "n/a" || got.CreatedUTC != "2026-09-11T00:00:00Z" ||
		got.Edges != 10 || got.EmbedDim != 4 || got.Steps != 3 {
		t.Fatalf("field mismatch: %+v", got)
	}
}

func TestHeaderRejectsBadVersion(t *testing.T) {
	if _, err := UnmarshalHeader([]byte(`{"format":"fly-reservoir","version":99}`)); err == nil {
		t.Fatal("expected version rejection")
	}
}

func TestHeaderRejectsBadFormat(t *testing.T) {
	if _, err := UnmarshalHeader([]byte(`{"format":"other","version":1}`)); err == nil {
		t.Fatal("expected format rejection")
	}
}

func TestHeaderRejectsUnknownField(t *testing.T) {
	if _, err := UnmarshalHeader([]byte(`{"format":"fly-reservoir","version":1,"surprise":1}`)); err == nil {
		t.Fatal("expected unknown-field rejection")
	}
}

func TestHeaderRejectsMissingRequiredFields(t *testing.T) {
	if _, err := UnmarshalHeader([]byte(`{"format":"fly-reservoir","version":1}`)); err == nil {
		t.Fatal("expected missing-fields rejection")
	}
}

func TestHeaderDecayExtensionRoundTrips(t *testing.T) {
	h := Header{Format: FormatString, Version: FormatVersion, Name: "d",
		Neurons: 4, Edges: 2, EmbedDim: 2, Steps: 2,
		Classes: []string{"x"}, WeightScale: 0.5, Decay: 0.7}
	b, err := MarshalHeader(h)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"decay":0.7`)) {
		t.Fatalf("decay key missing from JSON: %s", b)
	}
	got, err := UnmarshalHeader(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decay != 0.7 {
		t.Fatalf("decay = %v, want 0.7", got.Decay)
	}
}

func TestHeaderOmitsZeroDecay(t *testing.T) {
	h := Header{Format: FormatString, Version: FormatVersion, Name: "d",
		Neurons: 4, Edges: 2, EmbedDim: 2, Steps: 2,
		Classes: []string{"x"}, WeightScale: 0.5}
	b, err := MarshalHeader(h)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte("decay")) {
		t.Fatalf("unset decay must be omitted: %s", b)
	}
}

func TestHeaderRejectsOutOfRangeFields(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"zero neurons", `{"format":"fly-reservoir","version":1,"name":"n","neurons":0,"edges":1,"embed_dim":2,"steps":1,"classes":["a"],"weight_scale":0.5}`},
		{"negative edges", `{"format":"fly-reservoir","version":1,"name":"n","neurons":2,"edges":-1,"embed_dim":2,"steps":1,"classes":["a"],"weight_scale":0.5}`},
		{"zero steps", `{"format":"fly-reservoir","version":1,"name":"n","neurons":2,"edges":1,"embed_dim":2,"steps":0,"classes":["a"],"weight_scale":0.5}`},
		{"no classes", `{"format":"fly-reservoir","version":1,"name":"n","neurons":2,"edges":1,"embed_dim":2,"steps":1,"classes":[],"weight_scale":0.5}`},
		{"zero weight scale", `{"format":"fly-reservoir","version":1,"name":"n","neurons":2,"edges":1,"embed_dim":2,"steps":1,"classes":["a"],"weight_scale":0}`},
		{"decay out of range", `{"format":"fly-reservoir","version":1,"name":"n","neurons":2,"edges":1,"embed_dim":2,"steps":1,"classes":["a"],"weight_scale":0.5,"decay":1.5}`},
	}
	for _, tc := range cases {
		if _, err := UnmarshalHeader([]byte(tc.json)); err == nil {
			t.Fatalf("%s: expected rejection", tc.name)
		}
	}
}

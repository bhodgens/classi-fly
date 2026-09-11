package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

const fixtureFly = "../../tools/eval/eval_fixture.fly"

// requireFixture skips when the tiny fixture .fly is absent. The fixture is
// a build input from the offline tooling, not a source file, so tests degrade
// gracefully where it has not been generated.
func requireFixture(t *testing.T) string {
	t.Helper()
	if _, err := os.Stat(fixtureFly); err != nil {
		t.Skipf("fixture %s not present: %v", fixtureFly, err)
	}
	return fixtureFly
}

func TestServeLoadFailClosed(t *testing.T) {
	srv := NewServer()
	if got := srv.getRes(); got != nil {
		t.Fatalf("fresh server must hold no artifact, got %v", got)
	}
	err := srv.Load(filepath.Join(t.TempDir(), "missing.fly"))
	if err == nil {
		t.Fatal("Load on missing file must fail")
	}
	if got := srv.getRes(); got != nil {
		t.Fatalf("failed Load must leave server unloaded, got %v", got)
	}
}

func TestServeHealthz(t *testing.T) {
	srv := NewServer()
	handler := srv.Handler()

	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("/healthz before load: got %d, want 503", rr.Code)
	}
	if b := rr.Body.String(); b != "not loaded\n" {
		t.Fatalf("/healthz before load body: got %q", b)
	}

	if err := srv.Load(requireFixture(t)); err != nil {
		t.Fatalf("Load fixture: %v", err)
	}
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("/healthz after load: got %d, want 200", rr.Code)
	}
	if b := rr.Body.String(); b != "ok" {
		t.Fatalf("/healthz after load body: got %q, want \"ok\"", b)
	}
}

func TestServeInfo(t *testing.T) {
	srv := NewServer()
	handler := srv.Handler()

	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/info", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("/info before load: got %d, want 503", rr.Code)
	}

	if err := srv.Load(requireFixture(t)); err != nil {
		t.Fatalf("Load fixture: %v", err)
	}
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/info", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("/info after load: got %d, want 200", rr.Code)
	}
	var got map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("/info body not JSON: %v", err)
	}
	// Contract 4: /info exposes the header fields of the loaded artifact.
	if got["name"] != "eval-fixture" {
		t.Errorf("/info name: got %v, want eval-fixture", got["name"])
	}
	if got["format"] != "fly-reservoir" {
		t.Errorf("/info format: got %v, want fly-reservoir", got["format"])
	}
	if v, ok := got["embed_dim"].(float64); !ok || int(v) != 8 {
		t.Errorf("/info embed_dim: got %v (%T), want 8", got["embed_dim"], got["embed_dim"])
	}
	classes, ok := got["classes"].([]any)
	if !ok || len(classes) != 3 {
		t.Fatalf("/info classes: got %v (%T), want 3 entries", got["classes"], got["classes"])
	}
	for i, want := range []string{"alpha", "beta", "gamma"} {
		if classes[i] != want {
			t.Errorf("/info classes[%d]: got %v, want %s", i, classes[i], want)
		}
	}
}

func TestServeClassify(t *testing.T) {
	srv := NewServer()
	handler := srv.Handler()

	post := func(body []byte) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/classify", bytes.NewReader(body)))
		return rr
	}

	// Before load, /classify must refuse rather than misclassify.
	rr := post([]byte(`{"embedding":[1,0,0,0,0,0,0,0]}`))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("/classify before load: got %d, want 503", rr.Code)
	}

	if err := srv.Load(requireFixture(t)); err != nil {
		t.Fatalf("Load fixture: %v", err)
	}

	// Wrong-length embedding -> 400.
	rr = post([]byte(`{"embedding":[1,2,3]}`))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("wrong-length embedding: got %d, want 400 (body %s)", rr.Code, rr.Body.String())
	}

	// Invalid JSON -> 400.
	rr = post([]byte(`{"embedding":`))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("invalid JSON: got %d, want 400", rr.Code)
	}

	// Missing/empty embedding -> 400.
	rr = post([]byte(`{}`))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("empty embedding: got %d, want 400", rr.Code)
	}

	// One-hot basis vectors: the fixture's lane-c dominant projection makes
	// lane c win, so the class must equal the basis vector's name.
	cases := []struct {
		vec       []float32
		wantClass string
	}{
		{[]float32{1, 0, 0, 0, 0, 0, 0, 0}, "alpha"},
		{[]float32{0, 1, 0, 0, 0, 0, 0, 0}, "beta"},
		{[]float32{0, 0, 1, 0, 0, 0, 0, 0}, "gamma"},
	}
	for _, tc := range cases {
		body, err := json.Marshal(classifyRequest{Embedding: tc.vec})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		rr = post(body)
		if rr.Code != http.StatusOK {
			t.Fatalf("classify %v: got %d, want 200 (body %s)", tc.vec, rr.Code, rr.Body.String())
		}
		var resp classifyResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("classify body not JSON: %v", err)
		}
		if resp.Class != tc.wantClass {
			t.Errorf("classify %v: got class %q, want %q", tc.vec, resp.Class, tc.wantClass)
		}
		if resp.Abstained {
			t.Errorf("classify %v: fixture never abstains, got abstained=true", tc.vec)
		}
		if resp.Confidence < 0 || resp.Confidence > 1 {
			t.Errorf("classify %v: confidence %v out of 0..1", tc.vec, resp.Confidence)
		}
	}
}

func TestServeClassifyMethods(t *testing.T) {
	srv := NewServer()
	if err := srv.Load(requireFixture(t)); err != nil {
		t.Fatalf("Load fixture: %v", err)
	}
	handler := srv.Handler()

	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/classify", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /classify: got %d, want 405", rr.Code)
	}
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/info", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /info: got %d, want 405", rr.Code)
	}
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/healthz", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /healthz: got %d, want 405", rr.Code)
	}
}

func TestRunServeArgs(t *testing.T) {
	if code := runServe([]string{"--fly", fixtureFly, "extra"}); code != 1 {
		t.Errorf("positional args: got exit %d, want 1", code)
	}
	if code := runServe([]string{}); code != 1 {
		t.Errorf("missing --fly: got exit %d, want 1", code)
	}
	// Bad artifact path -> fail-closed exit 2 before any listener opens.
	if code := runServe([]string{"--fly", filepath.Join(t.TempDir(), "nope.fly")}); code != 2 {
		t.Errorf("missing artifact: got exit %d, want 2", code)
	}
}

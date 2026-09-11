// Sidecar HTTP face for classi-fly: `classi-fly serve` exposes a loaded .fly
// artifact over a minimal stdlib HTTP API (Contract 4). This file wraps the
// reservoir package; it never re-implements SpMV or the readout. See
// docs/INTEGRATION.md for the consumer-facing contract.

package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"sync"

	"github.com/caimlas/classi-fly/reservoir"
)

// defaultServeAddr is the loopback-only default listen address. The sidecar
// has no authentication, so it must never be exposed beyond loopback.
const defaultServeAddr = "127.0.0.1:8091"

// Server holds the sidecar state: an optional loaded artifact behind the
// three HTTP routes. The zero value answers 503 on /healthz until Load
// succeeds, which lets callers observe the load state directly.
type Server struct {
	mu  sync.RWMutex
	res *reservoir.Reservoir
}

// NewServer returns a Server with no artifact loaded (/healthz -> 503).
func NewServer() *Server {
	return &Server{}
}

// Load reads path through the reservoir package and installs it as the
// served artifact. It is fail-closed: on error the Server keeps its previous
// state and the caller must not serve.
func (s *Server) Load(path string) error {
	res, err := reservoir.Load(path)
	if err != nil {
		return fmt.Errorf("serve: load %s: %w", path, err)
	}
	s.mu.Lock()
	s.res = res
	s.mu.Unlock()
	return nil
}

// Handler returns the sidecar's HTTP mux: POST /classify, GET /info,
// GET /healthz.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/classify", s.handleClassify)
	mux.HandleFunc("/info", s.handleInfo)
	mux.HandleFunc("/healthz", s.handleHealthz)
	return mux
}

func (s *Server) getRes() *reservoir.Reservoir {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.res
}

type classifyRequest struct {
	Embedding []float32 `json:"embedding"`
	Trace     bool      `json:"trace"`
}

type classifyResponse struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	Margin     float64 `json:"margin"`
	Abstained  bool    `json:"abstained"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "method not allowed"})
		return
	}
	var req classifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid JSON body"})
		return
	}
	if len(req.Embedding) == 0 {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "embedding must be a non-empty array"})
		return
	}
	res := s.getRes()
	if res == nil {
		writeJSON(w, http.StatusServiceUnavailable, errorResponse{Error: "artifact not loaded"})
		return
	}
	out, err := res.Classify(req.Embedding)
	if err != nil {
		// The only runtime Classify failure is an input dimension mismatch;
		// a bad artifact can never be loaded, so this is a client error.
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, classifyResponse{
		Class:      out.Class,
		Confidence: out.Confidence,
		Margin:     out.Margin,
		Abstained:  out.Abstained,
	})
}

type infoJSON struct {
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

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "method not allowed"})
		return
	}
	res := s.getRes()
	if res == nil {
		writeJSON(w, http.StatusServiceUnavailable, errorResponse{Error: "artifact not loaded"})
		return
	}
	info := res.Info()
	writeJSON(w, http.StatusOK, infoJSON{
		Format:      "fly-reservoir",
		Version:     1,
		Name:        info.Name,
		Neurons:     info.Neurons,
		Edges:       info.Edges,
		EmbedDim:    info.EmbedDim,
		Steps:       info.Steps,
		Classes:     info.Classes,
		License:     info.License,
		Attribution: info.Attribution,
	})
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.getRes() == nil {
		http.Error(w, "not loaded", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// runServe implements the `classi-fly serve` subcommand: it loads the .fly
// artifact, then blocks serving the sidecar HTTP API on --addr. It returns an
// exit code (0 clean shutdown, 1 usage or bind error, 2 artifact load error)
// and never calls os.Exit; the CLI main maps the code.
func runServe(argv []string) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	addr := fs.String("addr", defaultServeAddr, "HTTP listen address (default is loopback only)")
	flyPath := fs.String("fly", "", "path to the .fly artifact (required)")
	if err := fs.Parse(argv); err != nil {
		return 1
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "serve: unexpected positional arguments")
		fmt.Fprintln(os.Stderr, "usage: classi-fly serve --fly model.fly [--addr 127.0.0.1:8091]")
		return 1
	}
	if *flyPath == "" {
		fmt.Fprintln(os.Stderr, "serve: --fly is required")
		fmt.Fprintln(os.Stderr, "usage: classi-fly serve --fly model.fly [--addr 127.0.0.1:8091]")
		return 1
	}
	srv := NewServer()
	// Fail-closed: the artifact is loaded before the listener opens, so a bad
	// artifact can never serve a classification. Bad artifact -> exit 2.
	if err := srv.Load(*flyPath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "serve: listen %s: %v\n", *addr, err)
		return 1
	}
	log.Printf("classi-fly serve: %s loaded, listening on %s (loopback only; do not expose)", *flyPath, ln.Addr())
	hs := &http.Server{Handler: srv.Handler()}
	if err := hs.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "serve: %v\n", err)
		return 1
	}
	return 0
}

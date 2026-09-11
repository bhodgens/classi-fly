# classi-fly Makefile - build, test, lint, fmt, release.
# The release target produces a stripped, trimpath'd static binary (no CGO).

GO ?= go
BINARY_NAME := classi-fly
BIN_DIR := bin
CMD := ./cmd/classi-fly
LDFLAGS := -s -w

.PHONY: all build test lint fmt release clean

all: build

build:
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN_DIR)/$(BINARY_NAME) $(CMD)

test:
	$(GO) test ./... -race -count=10

lint:
	$(GO) vet ./...
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then \
		echo "gofmt needed on:"; echo "$$out"; exit 1; fi

fmt:
	gofmt -w .

release: build
	@printf '%s\n' \
		"release binary: $(BIN_DIR)/$(BINARY_NAME)" \
		"size: $$(wc -c < $(BIN_DIR)/$(BINARY_NAME)) bytes"

clean:
	rm -rf $(BIN_DIR)

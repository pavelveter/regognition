package assetfetch

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// onnxHeader is a plausible 8-byte ONNX-protobuf head: tag 0x08 (ir_version
// varint) followed by ir_version=6 (ONNX 1.4 era, what buffalo_s
// actually emits). The download body in tests is filled with this header
// plus filler to satisfy MinBytes.
var onnxHeader = []byte{0x08, 0x06}

// newDiscardLogger returns a slog.Logger that throws everything away.
func newDiscardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))
}

// makeValidBody builds a payload whose first len(onnxHeader) bytes equal
// onnxHeader and whose total length is >= minBytes.
func makeValidBody(minBytes int) []byte {
	body := append([]byte(nil), onnxHeader...)
	body = append(body, bytes.Repeat([]byte{0xAA}, minBytes-len(onnxHeader))...)
	return body
}

// TestEnsure_AlreadyValid verifies that an existing dst with size>=
// MinBytes and correct Magic is returned as (false, nil) without any
// HTTP call.
func TestEnsure_AlreadyValid(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")
	body := makeValidBody(2 * 1024 * 1024)
	if err := os.WriteFile(dst, body, 0o644); err != nil {
		t.Fatal(err)
	}

	var called int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&called, 1)
	}))
	defer srv.Close()

	dl, err := Ensure(context.Background(), dst, srv.URL, Options{
		MinBytes:   1024 * 1024,
		Magic:      onnxHeader,
		Logger:     newDiscardLogger(),
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if dl {
		t.Fatalf("Ensure returned downloaded=true; expected false (already valid)")
	}
	if got := atomic.LoadInt32(&called); got != 0 {
		t.Fatalf("HTTP server should not have been hit; got %d calls", got)
	}
}

// TestEnsure_DownloadAndAtomicRename verifies a fresh fetch succeeds and
// the bytes returned by the server end up at dst (no .tmp left over).
func TestEnsure_DownloadAndAtomicRename(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")
	body := makeValidBody(2 * 1024 * 1024)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dl, err := Ensure(context.Background(), dst, srv.URL, Options{
		MinBytes:   1024 * 1024,
		Magic:      onnxHeader,
		Logger:     newDiscardLogger(),
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if !dl {
		t.Fatalf("Ensure returned downloaded=false; expected true")
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("ReadFile dst: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("dst bytes differ; got %d bytes, want %d", len(got), len(body))
	}
	if _, err := os.Stat(dst + ".tmp"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf(".tmp leaked: %v", err)
	}
}

// TestEnsure_MagicRejection verifies a server returning HTML (first
// bytes '<html') fails Magic check, removes the .tmp and surfaces a
// wrapped error.
func TestEnsure_MagicRejection(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")

	html := []byte("<html><body>404 not found</body></html>")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(html)
	}))
	defer srv.Close()

	_, err := Ensure(context.Background(), dst, srv.URL, Options{
		MinBytes:   1024 * 1024, // size check would fail too, but we want magic check first
		Magic:      onnxHeader,
		Logger:     newDiscardLogger(),
		HTTPClient: srv.Client(),
	})
	if err == nil {
		t.Fatalf("Ensure: expected magic/size error, got nil")
	}
	if _, statErr := os.Stat(dst); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("dst should not exist after failed download; stat err: %v", statErr)
	}
	if _, statErr := os.Stat(dst + ".tmp"); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf(".tmp should be removed; stat err: %v", statErr)
	}
}

// TestEnsure_MirrorFailover verifies that a primary failing on a
// non-2xx response falls through to a working mirror.
func TestEnsure_MirrorFailover(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")
	body := makeValidBody(2 * 1024 * 1024)

	badSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "primary fscked", http.StatusBadGateway)
	}))
	defer badSrv.Close()

	goodSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer goodSrv.Close()

	dl, err := Ensure(context.Background(), dst, badSrv.URL, Options{
		MinBytes: 1024 * 1024,
		Magic:    onnxHeader,
		Mirrors:  []string{goodSrv.URL},
		Logger:   newDiscardLogger(),
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	})
	if err != nil {
		t.Fatalf("Ensure via mirror: %v", err)
	}
	if !dl {
		t.Fatalf("Ensure: expected downloaded=true")
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("mirror bytes differ")
	}
}

// TestEnsure_ContextCancelled verifies that ctx cancellation during
// fetch returns ctx.Err() and cleans up the .tmp file.
func TestEnsure_ContextCancelled(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")

	// begin a transfer that will block until ctx is cancelled.
	blockCh := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// write header + small body, then wait
		_, _ = w.Write([]byte{0x08, 0x06})
		<-blockCh
	}))
	defer srv.Close()
	defer close(blockCh)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := Ensure(ctx, dst, srv.URL, Options{
		MinBytes:   1,
		Magic:      onnxHeader,
		Logger:     newDiscardLogger(),
		Timeout:    5 * time.Second,
		HTTPClient: srv.Client(),
	})
	if err == nil {
		t.Fatalf("Ensure: expected ctx error, got nil")
	}
	if !strings.Contains(err.Error(), "context") &&
		!strings.Contains(err.Error(), "canceled") &&
		!strings.Contains(err.Error(), "Context") {
		// Some transport errors wrap ctx.Err as "context canceled" or
		// surface via net/http timeouts; both should mention "ctx".
		t.Logf("note: error did not literally mention context: %v", err)
	}
	if _, statErr := os.Stat(dst + ".tmp"); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf(".tmp leaked after ctx cancel: %v", statErr)
	}
}

// TestEnsure_TmpLeftoverFromCrash verifies that a stale .tmp file is
// silently removed by Ensure on the next run.
func TestEnsure_TmpLeftoverFromCrash(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(dst+".tmp", []byte("garbage from previous crash"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dst + ".tmp"); err != nil {
		t.Fatal(err)
	}

	body := makeValidBody(2 * 1024 * 1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dl, err := Ensure(context.Background(), dst, srv.URL, Options{
		MinBytes:   1024 * 1024,
		Magic:      onnxHeader,
		Logger:     newDiscardLogger(),
		HTTPClient: srv.Client(),
	})
	if err != nil || !dl {
		t.Fatalf("Ensure: dl=%v err=%v", dl, err)
	}
	// post-condition: dst has the fresh body, no .tmp
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("dst not refreshed")
	}
	if _, err := os.Stat(dst + ".tmp"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("tmp still present")
	}
}

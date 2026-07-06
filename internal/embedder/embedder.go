// Package embedder is the single ML abstraction the pipeline depends on.
//
// A real implementation composes RetinaFace (face detection) + ArcFace
// (alignment + 512-d L2-normalized embedding). Detector+embedder are
// internal implementation details — callers only see one Extract method
// that yields one []float32 per detected face.
//
// The skeleton ships only a deterministic stub (stub.go) so the build is
// self-contained and the rest of the pipeline can be exercised without
// bundling the native libonnxruntime. Drop in a RetinaFace+ArcFace ONNX
// implementation later behind the same interface and nothing else needs
// to change.
package embedder

import (
	"context"
	"image"
)

// Face is the per-face payload a DebugSink receives.
//
// Aligned is the 112×112 ArcFace-aligned crop — the exact image the
// recognizer ingested. May be nil for embedders that cannot produce
// it (e.g. the cache wrapper on a hit, or implementations without a
// real alignment pass). Sinks that depend on it must check.
//
// BBox is in the source image's coordinate system (matching
// image.Bounds()). Score is the detector's confidence (0..1, 0 for
// implementations that don't compute one).
//
// Index is the face's position in the source image (0..N-1 in the
// order the embedder emitted faces). Use it for stable file names
// when serialising per-face debug output.
type Face struct {
	Embedding []float32
	BBox      image.Rectangle
	Aligned   image.Image
	Score     float32
	Index     int
}

// DebugSink is the per-face callback for the --debug mode. An
// implementation receives every detected face from one Extract call
// (sequentially, in emit order) plus the source path so it can
// derive a destination filename.
//
// Implementations MUST NOT assume OnFace is called from a single
// goroutine: a worker pool invokes Extract concurrently across
// images, so different OnFace invocations may overlap across
// goroutines. Concurrent calls for DIFFERENT srcPath values are
// safe (the sink just does file I/O with distinct paths). Concurrent
// calls for the SAME srcPath are NOT supported — the embedder
// guarantees no overlap per call.
type DebugSink interface {
	OnFace(ctx context.Context, srcPath string, face Face)
}

// Embedder extracts face embeddings (ArcFace-style, 512-d, L2-normalized).
//
// Methods MUST be safe for concurrent use across goroutines.
//
// Concurrency note for ONNX implementations: ONNX Runtime Session.Run()
// is technically thread-safe but contends on an internal mutex under
// heavy load. The recommended pattern is one session per worker
// goroutine — expose either an Embedder.NewSession() method (returning a
// fresh Embedder that shares loaded weights) or construct per-worker
// embedders in main.go. The skeleton stub is stateless, so this concern
// is moot today.
type Embedder interface {
	// Extract decodes/processors the image and returns one vector per face.
	Extract(ctx context.Context, img image.Image) ([][]float32, error)
	// ExtractFile is a convenience for callers that have only a path.
	ExtractFile(ctx context.Context, path string) ([][]float32, error)
	// ExtractWithDebug is like Extract but also emits per-face metadata
	// (bbox, 112×112 aligned crop, score) to sink. sink may be nil, in
	// which case the call is equivalent to Extract. When --debug is on
	// the pipeline uses this to capture face crops for matches; when
	// off it uses Extract. srcPath is propagated to the sink so file
	// naming can mirror the search-dir tree.
	ExtractWithDebug(ctx context.Context, img image.Image, srcPath string, sink DebugSink) ([][]float32, error)
	// Close releases any native resources (ONNX sessions, GPU memory, etc.).
	Close() error
}

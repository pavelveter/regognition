// batched_writer.go — buffers cache.Set calls and flushes them
// in batches from a single goroutine, so SQLite sees one commit
// per batch instead of one commit per photo.
//
// Integration: replace direct c.cache.Set(...) calls with
// w.Enqueue(...), and start the batchedWriter alongside cache.Open.
// Close it (which flushes any partial batch) before fc.Close().

package cache

import (
	"context"
	"log/slog"
	"time"
)

// pendingWrite is one queued cache row.
type pendingWrite struct {
	path  string
	hash  string
	faces [][]float32
}

// BatchedWriter buffers Set() calls and flushes them periodically (or
// when the buffer fills) from a single goroutine, so SQLite sees one
// commit per batch instead of one commit per photo.
type BatchedWriter struct {
	cache    FaceCache
	logger   *slog.Logger
	in       chan pendingWrite
	done     chan struct{}
	flushN   int
	flushDur time.Duration
}

// NewBatchedWriter starts the background flush loop. flushN=200,
// flushDur=1s are reasonable starting points: large enough to
// amortize commit overhead across a meaningful batch, small enough
// that a crash never loses more than ~1s of freshly-computed
// embeddings (which just get recomputed next run — never silently
// dropped, since ExtractFile already returned the correct result to
// the pipeline before the write was even enqueued).
func NewBatchedWriter(c FaceCache, logger *slog.Logger, flushN int, flushDur time.Duration) *BatchedWriter {
	if logger == nil {
		logger = slog.Default()
	}
	w := &BatchedWriter{
		cache:    c,
		logger:   logger,
		in:       make(chan pendingWrite, flushN*2),
		done:     make(chan struct{}),
		flushN:   flushN,
		flushDur: flushDur,
	}
	go w.loop()
	return w
}

// Enqueue is the non-blocking replacement for cache.Set. Best-effort,
// same as the direct Set call was: a full queue (writer stuck/slow)
// drops the write with a warning rather than blocking the calling
// pipeline worker — a cache miss next run is much cheaper than
// stalling image processing on cache I/O.
func (w *BatchedWriter) Enqueue(path, hash string, faces [][]float32) {
	select {
	case w.in <- pendingWrite{path: path, hash: hash, faces: faces}:
	default:
		w.logger.Warn("cache: write queue full, dropping entry", "path", path)
	}
}

func (w *BatchedWriter) loop() {
	batch := make([]pendingWrite, 0, w.flushN)
	ticker := time.NewTicker(w.flushDur)
	defer ticker.Stop()
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := w.flushBatch(batch); err != nil {
			w.logger.Warn("cache: batch flush failed", "count", len(batch), "err", err)
		}
		batch = batch[:0]
	}
	for {
		select {
		case pw, ok := <-w.in:
			if !ok {
				flush()
				close(w.done)
				return
			}
			batch = append(batch, pw)
			if len(batch) >= w.flushN {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

// flushBatch writes N rows in ONE transaction.
func (w *BatchedWriter) flushBatch(batch []pendingWrite) error {
	items := make([]BatchItem, len(batch))
	for i, pw := range batch {
		items[i] = BatchItem{Path: pw.path, Hash: pw.hash, Faces: pw.faces}
	}
	return w.cache.SetBatch(context.Background(), items)
}

// Close flushes any partial batch and waits for the writer goroutine
// to exit. Call before closing the underlying FaceCache.
func (w *BatchedWriter) Close() {
	close(w.in)
	<-w.done
}

// cached_embedder.go — decorator that wraps an embedder.Embedder
// with a path-keyed cache.

package cache

import (
	"context"
	"errors"
	"image"
	"log/slog"

	"regognition/internal/embedder"
)

// CachedEmbedder decorates an inner Embedder so that ExtractFile
// results transparently flow through a FaceCache:
//   - on hit and matching hash → return cached faces;
//   - on miss or stale hash     → inner.ExtractFile; if successful,
//     write to cache (best-effort).
//
// Extract(image.Image) is never cached: there is no stable key, and
// callers that hold an image.Image usually already have the original
// path if they cared about caching.
type CachedEmbedder struct {
	inner  embedder.Embedder
	cache  FaceCache
	logger *slog.Logger
}

// NewCached returns a CachedEmbedder wrapping inner around c. If
// logger is nil, slog.Default() is used.
func NewCached(inner embedder.Embedder, c FaceCache, logger *slog.Logger) *CachedEmbedder {
	if logger == nil {
		logger = slog.Default()
	}
	return &CachedEmbedder{inner: inner, cache: c, logger: logger}
}

// Extract always delegates to inner; no caching by image content.
func (c *CachedEmbedder) Extract(ctx context.Context, img image.Image) ([][]float32, error) {
	return c.inner.Extract(ctx, img)
}

// ExtractWithDebug delegates to inner. The cache does NOT short-circuit
// ExtractWithDebug: there is no stable key for an image.Image input
// (cache only keys by file path + content hash via ExtractFile), and
// the pipeline only ever calls ExtractWithDebug for pathed jobs, so
// pass-through is correct.
//
// Note: when the pipeline uses ExtractFile (which IS cache-keyed), it
// gets cached embeddings back without the sink being notified. main.go
// bypasses the cache entirely when --debug is on so this gap never
// affects operators; the pass-through here is a safety net for any
// future caller.
func (c *CachedEmbedder) ExtractWithDebug(ctx context.Context, img image.Image, srcPath string, sink embedder.DebugSink) ([][]float32, error) {
	return c.inner.ExtractWithDebug(ctx, img, srcPath, sink)
}

// ExtractFile looks up the file's hash in the cache; on miss it
// invokes the inner embedder and stores the result. Cache write
// failures are logged as warnings — they must NEVER mask the
// embedding result.
func (c *CachedEmbedder) ExtractFile(ctx context.Context, path string) ([][]float32, error) {
	hash, hashErr := HashFile(path)
	if hashErr != nil {
		// Likely a stat error — let the inner embedder attempt the
		// read; pipeline-level code will surface the real error.
		c.logger.Warn("cache: hash failed, skipping cache lookup", "path", path, "err", hashErr)
		return c.inner.ExtractFile(ctx, path)
	}

	storedHash, storedFaces, ok, lookupErr := c.cache.Get(ctx, path)
	if lookupErr != nil {
		// Treat any cache read failure as a miss to keep the pipeline
		// running even with a flaky cache.
		c.logger.Warn("cache: lookup failed, falling back to inner", "path", path, "err", lookupErr)
	} else if ok && storedHash == hash {
		return storedFaces, nil
	}

	faces, err := c.inner.ExtractFile(ctx, path)
	if err != nil {
		return nil, err
	}

	if err := c.cache.Set(ctx, path, hash, faces); err != nil {
		// Best-effort: log and return the embedding result anyway.
		c.logger.Warn("cache: write failed, returning fresh embeddings", "path", path, "err", err)
	}
	return faces, nil
}

// Close releases both the inner embedder and the underlying cache.
// Errors from each are joined (inner first).
func (c *CachedEmbedder) Close() error {
	return errors.Join(c.inner.Close(), c.cache.Close())
}

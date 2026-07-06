package pipeline

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"sync"

	"github.com/disintegration/imaging"

	"regognition/internal/cache"
)

// ImageJob is a decoded image sent from the Prefetcher to workers.
// Carries the path, decoded image (or nil for cache hits/errors),
// and any error from reading/decoding.
type ImageJob struct {
	Path     string
	Img      image.Image  // nil if cache hit or error
	Embeddings [][]float32 // non-nil if cache hit (pre-computed embeddings)
	Err      error        // non-nil if read/decode failed
}

// Prefetcher reads image files from disk in batches and sends
// decoded images to a channel for workers to process.
//
// On slow storage (USB/network HDD), sequential reads are 5-10x
// faster than random reads. The Prefetcher reads files in small
// batches (default 4-8) with parallel decode within each batch,
// keeping the disk busy while workers do CPU-bound inference.
//
// The Prefetcher also applies TargetDimension resize, avoiding
// redundant work in workers.
type Prefetcher struct {
	paths       []string
	batchSize   int
	targetDim   int
	outCh       chan ImageJob
	decodeWorkers int // goroutines per batch for parallel decode
	cache       cache.FaceCache // optional: skip I/O for cached images
	logger      *slog.Logger
}

// NewPrefetcher creates a Prefetcher that will read paths in
// batches of batchSize, decode images in parallel (decodeWorkers
// goroutines per batch), resize to targetDim if > 0, and send
// results to a buffered channel.
func NewPrefetcher(
	paths []string,
	batchSize int,
	targetDim int,
	decodeWorkers int,
	cache cache.FaceCache,
	logger *slog.Logger,
) *Prefetcher {
	if batchSize < 1 {
		batchSize = 4
	}
	if decodeWorkers < 1 {
		decodeWorkers = 2
	}
	if logger == nil {
		logger = slog.Default()
	}
	// Buffer enough to keep workers busy while disk reads happen.
	// 2x batchSize gives prefetcher a head start.
	bufSize := batchSize * 2
	if bufSize > len(paths) {
		bufSize = len(paths)
	}
	return &Prefetcher{
		paths:         paths,
		batchSize:     batchSize,
		targetDim:     targetDim,
		decodeWorkers: decodeWorkers,
		outCh:         make(chan ImageJob, bufSize),
		cache:         cache,
		logger:        logger,
	}
}

// Chan returns the output channel. Closed after all images are read.
func (p *Prefetcher) Chan() <-chan ImageJob {
	return p.outCh
}

// Run reads all paths in batches and sends decoded images to the
// output channel. Blocks until all files are processed or ctx is
// cancelled. Must be called in a goroutine (or before workers start).
func (p *Prefetcher) Run(ctx context.Context) {
	defer close(p.outCh)

	for start := 0; start < len(p.paths); start += p.batchSize {
		if ctx.Err() != nil {
			return
		}
		end := start + p.batchSize
		if end > len(p.paths) {
			end = len(p.paths)
		}
		batch := p.paths[start:end]
		p.readBatch(ctx, batch)
	}
}

// readBatch reads a batch of files in parallel, decodes them,
// resizes if needed, and sends results to the output channel.
func (p *Prefetcher) readBatch(ctx context.Context, batch []string) {
	results := make([]result, len(batch))
	var wg sync.WaitGroup

	// Limit concurrent decode workers to avoid disk thrashing
	// on slow storage. More than 2-3 parallel reads on a USB
	// drive actually hurts throughput due to seeking.
	sem := make(chan struct{}, p.decodeWorkers)

	for i, path := range batch {
		wg.Add(1)
		go func(idx int, filePath string) {
			defer wg.Done()

			// Acquire semaphore
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				results[idx] = result{path: filePath, err: ctx.Err()}
				return
			}

			results[idx] = p.readAndResize(ctx, filePath)
		}(i, path)
	}

	wg.Wait()

	// Send results in order to preserve deterministic output
	for _, r := range results {
		select {
		case p.outCh <- ImageJob{
			Path:       r.path,
			Img:        r.img,
			Embeddings: r.embeddings,
			Err:        r.err,
		}:
		case <-ctx.Done():
			return
		}
	}
}

// readAndResize reads a single image file, decodes it, and applies
// TargetDimension resize if configured. Checks cache first — on hit,
// skips file read entirely.
func (p *Prefetcher) readAndResize(ctx context.Context, path string) result {
	if ctx.Err() != nil {
		return result{path: path, err: ctx.Err()}
	}

	// Check cache before reading file. HashFile uses os.Stat
	// (mtime + size) — fast on any storage.
	if p.cache != nil {
		hash, hashErr := cache.HashFile(path)
		if hashErr == nil {
			faces, ok, lookupErr := p.cache.Has(ctx, path, hash)
			if lookupErr != nil {
				p.logger.Warn("cache: lookup failed in prefetcher", "path", path, "err", lookupErr)
			} else if ok {
				p.logger.Debug("cache hit in prefetcher", "path", path, "faces", len(faces))
				return result{path: path, embeddings: faces}
			}
		}
	}

	// Cache miss — read and decode file
	img, err := imaging.Open(path, imaging.AutoOrientation(true))
	if err != nil {
		return result{path: path, err: fmt.Errorf("read: %w", err)}
	}

	if p.targetDim > 0 {
		b := img.Bounds()
		if b.Dx() > p.targetDim || b.Dy() > p.targetDim {
			img = imaging.Fit(img, p.targetDim, p.targetDim, imaging.CatmullRom)
		}
	}

	return result{path: path, img: img}
}

// result is an internal type for readAndResize return values.
type result struct {
	path       string
	img        image.Image
	embeddings [][]float32
	err        error
}

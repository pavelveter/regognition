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
	Path       string
	Img        image.Image // nil if cache hit or error
	Embeddings [][]float32 // non-nil if cache hit (pre-computed embeddings)
	Err        error       // non-nil if read/decode failed
}

// Prefetcher reads image files from disk/network in parallel and
// streams decoded images to a channel as each one finishes — no
// batch synchronization, so a single slow read never blocks results
// that are already ready.
//
// ReadWorkers controls how many files can be in flight at once.
// Tune this to your storage:
//   - Local SSD / typical network share (SMB/NFS over LAN): high
//     concurrency hides per-request latency well. Start at 2-4x your
//     CPU count and increase while watching throughput/error rate.
//   - Spinning disk / USB HDD: physical seek cost makes high
//     concurrency counterproductive. 2-3 is often the sweet spot.
//   - There is no one right default — that's WHY this is a config
//     knob now instead of a hardcoded constant.
type Prefetcher struct {
	paths       []string
	targetDim   int
	readWorkers int
	outCh       chan ImageJob
	cache       cache.FaceCache
	logger      *slog.Logger
}

// NewPrefetcher creates a Prefetcher. readWorkers < 1 defaults to 4
// (a conservative middle ground; override via config for your actual
// storage medium — see Prefetcher doc comment).
func NewPrefetcher(
	paths []string,
	targetDim int,
	readWorkers int,
	cache cache.FaceCache,
	logger *slog.Logger,
) *Prefetcher {
	if readWorkers < 1 {
		readWorkers = 4
	}
	if logger == nil {
		logger = slog.Default()
	}
	bufSize := readWorkers * 2
	if bufSize > len(paths) {
		bufSize = len(paths)
	}
	if bufSize < 1 {
		bufSize = 1
	}
	return &Prefetcher{
		paths:       paths,
		targetDim:   targetDim,
		readWorkers: readWorkers,
		outCh:       make(chan ImageJob, bufSize),
		cache:       cache,
		logger:      logger,
	}
}

// Chan returns the output channel. Closed after all images are read.
func (p *Prefetcher) Chan() <-chan ImageJob {
	return p.outCh
}

// Run fans out ReadWorkers goroutines pulling from a shared path
// channel and streams each ImageJob to outCh the instant its own
// read finishes — no waiting on siblings, no batch boundaries.
// Blocks until all paths are processed or ctx is cancelled. Must be
// called in a goroutine (or before workers start draining Chan()).
func (p *Prefetcher) Run(ctx context.Context) {
	defer close(p.outCh)

	pathCh := make(chan string, p.readWorkers)
	go func() {
		defer close(pathCh)
		for _, path := range p.paths {
			select {
			case pathCh <- path:
			case <-ctx.Done():
				return
			}
		}
	}()

	var wg sync.WaitGroup
	for i := 0; i < p.readWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range pathCh {
				r := p.readAndResize(ctx, path)
				select {
				case p.outCh <- ImageJob{Path: r.path, Img: r.img, Embeddings: r.embeddings, Err: r.err}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	wg.Wait()
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

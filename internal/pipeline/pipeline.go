// Package pipeline runs the worker-pool channel pipeline described in
// TODO.md:
//
//	[scan] ─paths──> [workers: decode+infer+match] ─outcomes──> [collect+copy]
//
// Errors per-job do NOT abort the pipeline; they are surfaced via
// Outcome.Err and counted in Stats.
package pipeline

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/disintegration/imaging"

	"regognition/internal/cache"
	"regognition/internal/cosine"
	"regognition/internal/embedder"
	"regognition/internal/persona"
	"regognition/internal/scanner"
)

// Job is one image submitted to the worker pool.
type Job struct {
	Path string
	// Img is the pre-decoded image from the Prefetcher.
	// When non-nil, workers skip imaging.Open and go straight
	// to inference. When nil (e.g. cache hit), workers should
	// have cached embeddings available.
	Img image.Image
}

// Outcome is the result for one image.
type Outcome struct {
	Path    string
	Matched bool
	Err     error
}

// Stats is a thread-safe running summary of pipeline activity.
//
// Always pass *Stats; copying after first use would break atomic fields.
type Stats struct {
	Scanned atomic.Int64
	Matched atomic.Int64
	Errors  atomic.Int64
	// Skipped counts outcomes that arrived after the collect loop
	// broke early on ctx (Ctrl-C / SIGTERM). Workers have already
	// stopped by then, so these are buffered matches that the
	// operator intentionally did not copy to finded/.
	Skipped atomic.Int64
}

// Options bundles everything the pipeline needs.
//
// CopyFile is overridable so tests can intercept without touching disk.
//
// TargetDimension, when > 0, caps each archive photo's longest edge
// BEFORE handing it to the Embedder. The image is fit (NOT stretched)
// so the aspect ratio is preserved: a 4000x3000 landscape becomes
// 1280x960 when TargetDimension=1280; a 3000x4000 portrait becomes
// 960x1280. Smaller photos pass through untouched.
//
// DebugSink, when non-nil, switches processJob from Extract to
// ExtractWithDebug so per-face metadata (112×112 aligned crop, bbox,
// score) is emitted to the sink alongside the standard embeddings.
// The pipeline itself still does the matching; sinks that care
// about face crops (FileDebugSink) re-derive dist from
// Persona.Embeddings. The default zero value (nil) keeps the
// non-debug path zero-overhead — the only extra cost is one
// nil-check per face.
type Options struct {
	Workers           int
	PrefetchBatchSize int // files to read ahead (default 4-8)
	OutputDir         string
	TargetDimension   int
	Threshold         float32
	Embedder          embedder.Embedder
	Persona           *persona.Persona
	Cache             cache.FaceCache      // optional: for prefetch cache integration
	Writer            *cache.BatchedWriter // optional: batched cache writer (replaces direct Cache.Set)
	DirSkip           string               // comma-separated folder names to skip
	SkipFiles         string               // comma-separated glob patterns for files to skip
	Extensions        string               // comma-separated allowed extensions (empty = default)
	Logger            *slog.Logger
	Stats             *Stats
	CopyFile          func(src, dst string) error
	DebugSink         embedder.DebugSink
}

// Run scans dir, fans out Workers goroutines, and writes matches
// into OutputDir preserving the archive's subdir tree (see
// filepath.Rel below). Returns nil on a clean run with per-job
// errors counted in Stats; returns an error only for setup failures.
func Run(ctx context.Context, dir string, opt Options) error {
	if opt.Workers <= 0 {
		return errors.New("pipeline: workers must be positive")
	}
	if opt.Embedder == nil {
		return errors.New("pipeline: nil embedder")
	}
	if opt.Persona == nil || len(opt.Persona.Embeddings) == 0 {
		return errors.New("pipeline: persona with embeddings required")
	}
	if opt.Logger == nil {
		opt.Logger = slog.Default()
	}
	if opt.Stats == nil {
		opt.Stats = &Stats{}
	}
	if opt.CopyFile == nil {
		opt.CopyFile = defaultCopy
	}
	if info, statErr := os.Stat(opt.OutputDir); statErr == nil && !info.IsDir() {
		return fmt.Errorf("pipeline: output path exists but is not a directory: %q", opt.OutputDir)
	}
	if err := os.MkdirAll(opt.OutputDir, 0o755); err != nil {
		return fmt.Errorf("pipeline: create output dir: %w", err)
	}

	// Normalize the search dir to an absolute path BEFORE handing it
	// to scanner.Walk and BEFORE using it as the base for filepath.Rel.
	// scanner.Walk returns absolute paths; if dir is given relatively
	// (e.g. via --search ./archive) filepath.Rel produces paths
	// starting with "..", tripping the fallback-to-basename guard
	// and silently flattening the finded/ tree. Normalizing first
	// keeps subdirs intact even for relative inputs.
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("pipeline: resolve absolute search dir: %w", err)
	}
	dir = absDir

	// Parse dir_skip: comma-separated list of folder names to skip.
	var skipDirs []string
	if opt.DirSkip != "" {
		for _, s := range strings.Split(opt.DirSkip, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				skipDirs = append(skipDirs, s)
			}
		}
		if len(skipDirs) > 0 {
			opt.Logger.Info("skipping directories", "dirs", skipDirs)
		}
	}

	// Parse skip_files: comma-separated glob patterns for files to skip.
	var skipFiles []string
	if opt.SkipFiles != "" {
		for _, s := range strings.Split(opt.SkipFiles, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				skipFiles = append(skipFiles, s)
			}
		}
		if len(skipFiles) > 0 {
			opt.Logger.Info("skipping file patterns", "patterns", skipFiles)
		}
	}

	// Parse extensions: comma-separated allowed extensions.
	var extensions []string
	if opt.Extensions != "" {
		for _, s := range strings.Split(opt.Extensions, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				// Normalize: ensure leading dot, lowercase.
				if !strings.HasPrefix(s, ".") {
					s = "." + s
				}
				s = strings.ToLower(s)
				extensions = append(extensions, s)
			}
		}
		if len(extensions) > 0 {
			opt.Logger.Info("file extensions filter", "extensions", extensions)
		}
	}

	paths, err := scanner.WalkWithOptions(dir, scanner.WalkOptions{
		SkipDirs:   skipDirs,
		SkipFiles:  skipFiles,
		Extensions: extensions,
	})
	if err != nil {
		return fmt.Errorf("pipeline: scan %q: %w", dir, err)
	}
	opt.Logger.Info("scan done", "dir", dir, "images", len(paths))

	if len(paths) == 0 {
		opt.Logger.Warn("no images in search dir", "dir", dir)
		return nil
	}

	// Stage: prefetcher reads files in batches, decodes images,
	// and sends them to workers. On slow storage (USB/network HDD),
	// sequential reads are 5-10x faster than random reads.
	batchSize := opt.PrefetchBatchSize
	if batchSize < 1 {
		batchSize = 4
	}
	prefetcher := NewPrefetcher(paths, batchSize, opt.TargetDimension, 2, opt.Cache, opt.Logger)

	// Stage: fan-out workers (CPU only, no disk I/O).
	outcomes := make(chan Outcome, len(paths))

	var wg sync.WaitGroup
	for w := 0; w < opt.Workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for job := range prefetcher.Chan() {
				if ctx.Err() != nil {
					outcomes <- Outcome{Path: job.Path, Err: ctx.Err()}
					break
				}
				outcomes <- processImageJob(ctx, job, opt)
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(outcomes)
	}()

	// Start prefetcher in background (reads files + decode)
	go func() {
		prefetcher.Run(ctx)
	}()

	// Stage: batch progress reporter. Emits a periodic slog.Info line
	// summarising scanned / matched / errors counters so the operator
	// can see liveness during long runs without enabling Debug.
	progressDone := make(chan struct{})
	go func(total int) {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-progressDone:
				return
			case <-ticker.C:
				opt.Logger.Info("batch progress",
					"total", total,
					"scanned", opt.Stats.Scanned.Load(),
					"matched", opt.Stats.Matched.Load(),
					"errors", opt.Stats.Errors.Load(),
				)
			}
		}
	}(len(paths))

	// Stage: collect and copy matches, preserving subdir structure.
	//
	// Honour ctx here: after SIGINT/SIGTERM, workers stop quickly but
	// the outcomes channel still has buffered matches in it. Without
	// this check, a Ctrl-C in the middle of a long run would still
	// copy every remaining buffered match to finded/ — the operator
	// would see N files land after the "interrupt received" line.
	for o := range outcomes {
		if ctx.Err() != nil {
			// Count this outcome AND drain the rest of the buffered
			// ones as skipped. Workers have already broken on
			// ctx.Err() and the closer goroutine will close
			// (outcomes) once wg.Wait() returns, so the drain
			// terminates without blocking.
			opt.Stats.Skipped.Add(1)
			for range outcomes {
				opt.Stats.Skipped.Add(1)
			}
			opt.Logger.Warn("shutdown: aborting collect phase",
				"skipped", opt.Stats.Skipped.Load(),
				"copied_before_shutdown", opt.Stats.Matched.Load())
			break
		}
		if o.Err != nil {
			opt.Stats.Errors.Add(1)
			opt.Logger.Warn("job error", "path", o.Path, "err", o.Err)
			continue
		}
		opt.Stats.Scanned.Add(1)
		if !o.Matched {
			continue
		}
		opt.Stats.Matched.Add(1)
		rel, relErr := filepath.Rel(dir, o.Path)
		if relErr != nil || strings.HasPrefix(rel, "..") {
			// Fallback: the photo lives outside the search root. Just
			// drop it under OutputDir by basename.
			rel = filepath.Base(o.Path)
		}
		dst := filepath.Join(opt.OutputDir, rel)
		if err := opt.CopyFile(o.Path, dst); err != nil {
			opt.Logger.Error("copy failed", "src", o.Path, "dst", dst, "err", err)
			continue
		}
		opt.Logger.Info("match copied", "src", o.Path, "dst", dst)
	}
	close(progressDone)

	opt.Logger.Info("pipeline done",
		"scanned", opt.Stats.Scanned.Load(),
		"matched", opt.Stats.Matched.Load(),
		"errors", opt.Stats.Errors.Load(),
		"skipped", opt.Stats.Skipped.Load(),
	)
	return nil
}

// processJob is the per-image workload: read + (optional) max-edge
// resize + infer + match.
//
// If a single face in the archive image is close enough to ANY persona
// embedding (cosine distance <= threshold), the photo is considered a
// match and the rest of the faces are not consulted (short-circuit).
//
// The resize uses imaging.Fit(img, d, d, ...) so the longest edge of
// any orientation (portrait OR landscape) lands on opt.TargetDimension
// while aspect ratio is preserved. CatmullRom is the downscale filter —
// a good accuracy/speed balance for face-recognizer preprocessing.
func processJob(ctx context.Context, job Job, opt Options) Outcome {
	img, err := imaging.Open(job.Path, imaging.AutoOrientation(true))
	if err != nil {
		opt.Logger.Debug("read failed", "path", job.Path, "err", err)
		return Outcome{Path: job.Path, Err: fmt.Errorf("read: %w", err)}
	}
	if opt.TargetDimension > 0 {
		b := img.Bounds()
		if b.Dx() > opt.TargetDimension || b.Dy() > opt.TargetDimension {
			// imaging.Fit preserves aspect ratio: portrait and
			// landscape frames both end up with the longest edge
			// exactly at TargetDimension. CatmullRom is a good
			// downscale filter (sharper than Linear, no ringing
			// artifacts like Lanczos on text/edges).
			img = imaging.Fit(img, opt.TargetDimension, opt.TargetDimension, imaging.CatmullRom)
		}
	}
	var faces [][]float32
	if opt.DebugSink != nil {
		faces, err = opt.Embedder.ExtractWithDebug(ctx, img, job.Path, opt.DebugSink)
	} else {
		faces, err = opt.Embedder.Extract(ctx, img)
	}
	if err != nil {
		opt.Logger.Debug("extract failed", "path", job.Path, "err", err)
		return Outcome{Path: job.Path, Err: fmt.Errorf("extract: %w", err)}
	}
	opt.Logger.Debug("extract done", "path", job.Path, "faces", len(faces))
	if len(faces) == 0 {
		return Outcome{Path: job.Path}
	}
	for _, f := range faces {
		dist, idx := cosine.BestMatch(f, opt.Persona.Embeddings)
		opt.Logger.Debug("best distance", "path", job.Path, "dist", dist, "persona_idx", idx)
		if dist <= opt.Threshold {
			opt.Logger.Info("match", "path", job.Path, "dist", dist)
			return Outcome{Path: job.Path, Matched: true}
		}
	}
	return Outcome{Path: job.Path}
}

// processImageJob is like processJob but accepts a pre-decoded image
// from the Prefetcher. The image is already resized to TargetDimension.
// This function does ONLY inference + matching (no disk I/O).
func processImageJob(ctx context.Context, job ImageJob, opt Options) Outcome {
	if job.Err != nil {
		opt.Logger.Debug("prefetch failed", "path", job.Path, "err", job.Err)
		return Outcome{Path: job.Path, Err: job.Err}
	}

	// Cache hit: embeddings were retrieved by the prefetcher.
	// Skip inference entirely — just match against persona.
	if len(job.Embeddings) > 0 {
		opt.Logger.Debug("cache hit in worker", "path", job.Path, "faces", len(job.Embeddings))
		for _, f := range job.Embeddings {
			dist, idx := cosine.BestMatch(f, opt.Persona.Embeddings)
			opt.Logger.Debug("best distance", "path", job.Path, "dist", dist, "persona_idx", idx)
			if dist <= opt.Threshold {
				opt.Logger.Info("match", "path", job.Path, "dist", dist)
				return Outcome{Path: job.Path, Matched: true}
			}
		}
		return Outcome{Path: job.Path}
	}

	// Cache miss: run inference on decoded image
	if job.Img == nil {
		return Outcome{Path: job.Path}
	}
	var faces [][]float32
	var err error
	if opt.DebugSink != nil {
		faces, err = opt.Embedder.ExtractWithDebug(ctx, job.Img, job.Path, opt.DebugSink)
	} else {
		faces, err = opt.Embedder.Extract(ctx, job.Img)
	}
	if err != nil {
		opt.Logger.Debug("extract failed", "path", job.Path, "err", err)
		return Outcome{Path: job.Path, Err: fmt.Errorf("extract: %w", err)}
	}

	// Store embeddings in cache for future runs (best-effort).
	// Use the batched writer if available; fall back to direct Set.
	if len(faces) > 0 {
		hash, hashErr := cache.HashFile(job.Path)
		if hashErr == nil {
			if opt.Writer != nil {
				opt.Writer.Enqueue(job.Path, hash, faces)
			} else if opt.Cache != nil {
				if cacheErr := opt.Cache.Set(ctx, job.Path, hash, faces); cacheErr != nil {
					opt.Logger.Debug("cache: write failed", "path", job.Path, "err", cacheErr)
				}
			}
		}
	}

	opt.Logger.Debug("extract done", "path", job.Path, "faces", len(faces))
	if len(faces) == 0 {
		return Outcome{Path: job.Path}
	}
	for _, f := range faces {
		dist, idx := cosine.BestMatch(f, opt.Persona.Embeddings)
		opt.Logger.Debug("best distance", "path", job.Path, "dist", dist, "persona_idx", idx)
		if dist <= opt.Threshold {
			opt.Logger.Info("match", "path", job.Path, "dist", dist)
			return Outcome{Path: job.Path, Matched: true}
		}
	}
	return Outcome{Path: job.Path}
}

// defaultCopy streams src -> dst, creating the destination's parent
// directory tree (preserving archive subdirs) if it doesn't exist yet.
func defaultCopy(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	// dst may include subdirectories under OutputDir (e.g.
	// `finded/2. Для печати и дизайна/foo.jpg`); os.Create alone
	// fails with "no such file or directory" if the parent is
	// missing. MkdirAll is idempotent, so existing dirs are a no-op.
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("create dst parent: %w", err)
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

// FileDebugSink is the canonical embedder.DebugSink for --debug mode.
// For every OnFace call it re-derives the cosine distance against the
// persona; if the face is a match (dist <= Threshold) and the embedder
// provided an Aligned image, it saves that image as a JPEG under
// DebugDir, mirroring SearchDir's subdir tree. Faces without an
// Aligned image (cache hits, stub-without-content) are silently
// skipped — the operator accepted that trade-off when --debug was
// combined with the cache in main.go (bypassed).
//
// FileDebugSink is safe for concurrent OnFace calls across DIFFERENT
// srcPath values (each sink call writes to a unique dst path). It is
// NOT safe for concurrent OnFace calls for the SAME srcPath — the
// embedder guarantees no overlap per Extract, so this never happens
// in practice.
//
// The persona reference is held by pointer; the caller (main.go)
// must keep it alive for the pipeline's lifetime. Threshold and the
// persona's embeddings are read-only after construction.
type FileDebugSink struct {
	// SearchDir is the absolute search dir the pipeline was given
	// (after filepath.Abs in Run). OnFace uses it as the base for
	// filepath.Rel when computing the mirror subdir.
	SearchDir string
	// DebugDir is the base for saved crops, typically
	// <OutputDir>/debug. It is created lazily by MkdirAll on the
	// first save — no setup required from the caller.
	DebugDir string
	// Threshold is the same cosine-distance cutoff the pipeline
	// uses for matching. Faces with dist > Threshold are not
	// saved; the pipeline's own short-circuit still produces the
	// Outcome.Matched flag, so the operator gets one
	// match-copied line per saved crop.
	Threshold float32
	// Persona is the flat list of L2-normalized embeddings. The
	// sink re-derives dist with cosine.BestMatch for every face.
	Persona [][]float32
	// Logger receives a Debug line per saved crop (and per
	// skipped face with no Aligned image). nil falls back to
	// slog.Default().
	Logger *slog.Logger
}

// OnFace implements embedder.DebugSink. See FileDebugSink for the
// concurrency contract and filename layout.
func (s *FileDebugSink) OnFace(ctx context.Context, srcPath string, f embedder.Face) {
	// Respect ctx cancellation so the sink's output stays consistent
	// with the collect loop: if SIGINT landed mid-Extract, the
	// collect loop will drop the matched outcome (so no photo lands
	// in finded/), and we should drop the matching crop too. Without
	// this check the operator sees orphan debug crops with no
	// parent photo in finded/ after a Ctrl-C. See the previous
	// code-review for the trade-off analysis.
	if ctx.Err() != nil {
		return
	}
	logger := s.Logger
	if logger == nil {
		logger = slog.Default()
	}
	// Embedders that can't produce Aligned (cache hit, stub without
	// image content) are skipped silently — the user explicitly
	// bypassed the cache when --debug is on, so this branch
	// signals an unexpected implementation rather than a real
	// failure.
	if f.Aligned == nil {
		logger.Debug("debug sink: face without Aligned image, skipping",
			"path", srcPath, "face_index", f.Index)
		return
	}
	dist, _ := cosine.BestMatch(f.Embedding, s.Persona)
	if dist > s.Threshold {
		// Not a match: silently skip. The pipeline's per-image
		// match decision is separate and authoritative; the
		// sink exists only to capture crops for matches.
		return
	}
	// Mirror the search dir structure under DebugDir. rel is the
	// path of the source photo relative to SearchDir, so
	// e.g. `2. Для печати и дизайна/foo.jpg`. Fall back to the
	// basename if the photo lives outside the search root.
	rel, err := filepath.Rel(s.SearchDir, srcPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		rel = filepath.Base(srcPath)
	}
	// Strip the original extension and append the face index +
	// dist. %%.3f gives 3 decimals (matches cluster around
	// 0.30-0.45; one decimal is too coarse for tuning, four
	// is noise).
	ext := filepath.Ext(rel)
	stem := strings.TrimSuffix(rel, ext)
	dst := filepath.Join(s.DebugDir, fmt.Sprintf("%s_face%d_dist%.3f.jpg", stem, f.Index, dist))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		logger.Warn("debug sink: mkdir failed", "dst", dst, "err", err)
		return
	}
	// imaging.Save picks JPEG from the .jpg extension; quality 95
	// is the library default and is plenty for 112×112 face
	// crops (any higher would just bloat without visual benefit).
	if err := imaging.Save(f.Aligned, dst); err != nil {
		logger.Warn("debug sink: save failed", "dst", dst, "err", err)
		return
	}
	logger.Info("debug crop saved",
		"src", srcPath, "dst", dst, "dist", dist, "face_index", f.Index)
}

// PersonaDebugSink is the --debug sink for persona selfies. Unlike
// FileDebugSink (which re-derives cosine distance against the
// persona embeddings and saves ONLY matches), PersonaDebugSink saves
// every detected face unconditionally — the persona selfies are
// themselves the reference set, so filtering them against
// themselves makes no sense. The output is sub-directory
// `<DebugDir>/persona/<rel-stem>_face<N>.jpg` (no `_dist<X>`
// suffix), distinct from FileDebugSink's `<DebugDir>/<rel-stem>_...`
// filename layout so the operator can see at a glance which output
// came from the persona extraction pass vs the archive match pass.
//
// PersonaDebugSink is safe for concurrent OnFace across DIFFERENT
// srcPath values, matching the FileDebugSink contract (the embedder
// guarantees no overlap per Extract call). Concurrent OnFace for
// the SAME srcPath is not supported.
//
// SourceDir is held by value; the caller (main.go) must keep cfg.PersonaDir
// alive while the persona extraction pass is in flight.
type PersonaDebugSink struct {
	// SourceDir is the persona dir (absolute if possible) used as
	// the base for filepath.Rel when computing the mirror subdir
	// under DebugDir/persona.
	SourceDir string
	// DebugDir is the base; the crops land under `<DebugDir>/persona/`.
	DebugDir string
	// Logger receives a Debug line per saved crop (and per skipped
	// face without an Aligned image). nil falls back to slog.Default().
	Logger *slog.Logger
}

// OnFace implements embedder.DebugSink. Saves every face with a
// non-nil Aligned image; silently skips cache hits and stub-without-
// implementation cases (no Aligned = nothing useful to save).
// No cosine distance is computed — persona selfies ARE the
// reference, so a threshold would only suppress signal.
func (s *PersonaDebugSink) OnFace(_ context.Context, srcPath string, f embedder.Face) {
	logger := s.Logger
	if logger == nil {
		logger = slog.Default()
	}
	if f.Aligned == nil {
		logger.Debug("persona debug sink: face without Aligned image, skipping",
			"path", srcPath, "face_index", f.Index)
		return
	}
	// Mirror the persona-dir structure under <DebugDir>/persona.
	// e.g. selfie at `persona/2026.../foo.jpg` -> `<DebugDir>/persona/2026.../foo_face0.jpg`.
	rel, err := filepath.Rel(s.SourceDir, srcPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		rel = filepath.Base(srcPath)
	}
	ext := filepath.Ext(rel)
	stem := strings.TrimSuffix(rel, ext)
	dst := filepath.Join(s.DebugDir, "persona", fmt.Sprintf("%s_face%d.jpg", stem, f.Index))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		logger.Warn("persona debug sink: mkdir failed", "dst", dst, "err", err)
		return
	}
	if err := imaging.Save(f.Aligned, dst); err != nil {
		logger.Warn("persona debug sink: save failed", "dst", dst, "err", err)
		return
	}
	logger.Info("persona crop saved",
		"src", srcPath, "dst", dst, "face_index", f.Index, "score", f.Score)
}

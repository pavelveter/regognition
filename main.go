// Command regognition finds photos of a specific person inside a local
// photo archive.
//
// Usage:
//
//	regognition [-config config.ini] [-search DIR] [-out DIR] [-persona DIR]
//
// Default flow:
//  1. Load config.ini (paths/persona, dir_search, dir_finded, ml.*, pipeline.*).
//  2. Apply CLI overrides (always win over INI).
//  3. Walk the persona directory and extract a flat list of ArcFace-style
//     512-d L2-normalized vectors from every detected face.
//  4. Walk the search directory, fan out workers, infer each photo, and
//     copy photos whose best-distance face matches the persona under the
//     configured threshold into the output directory.
//
// The skeleton ships a deterministic stub embedder; swap embedder.NewStub
// for an ONNX-backed implementation without changing pipeline code.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"time"

	"regognition/internal/assetfetch"
	"regognition/internal/cache"
	"regognition/internal/config"
	"regognition/internal/embedder"
	"regognition/internal/persona"
	"regognition/internal/pipeline"
	"regognition/internal/prettylog"
)

// buildEmbedder chooses between the ONNX-backed embedder and the
// deterministic stub. ONNX is preferred when the config points at
// the native ORT lib AND both model files exist on disk; otherwise
// the stub is used (with a warning so the fallback is never silent).
func buildEmbedder(cfg *config.Config, logger *slog.Logger) (embedder.Embedder, string) {
	missing := false
	for _, p := range []string{cfg.DetectorModelPath, cfg.EmbedderModelPath} {
		if _, err := os.Stat(p); err != nil {
			missing = true
			logger.Warn("embedder model missing, falling back to stub", "path", p)
		}
	}
	if cfg.OrtLibPath == "" {
		logger.Warn("ort_lib not set, falling back to stub", "hint", "see config.ini.example [ml].ort_lib")
		return embedder.NewStub(), "stub"
	}
	if _, err := os.Stat(cfg.OrtLibPath); err != nil {
		logger.Warn("ort_lib not found, falling back to stub", "path", cfg.OrtLibPath)
		return embedder.NewStub(), "stub"
	}
	if missing {
		return embedder.NewStub(), "stub"
	}
	emb, err := embedder.NewONNX(cfg.DetectorModelPath, cfg.EmbedderModelPath, embedder.ONNXOptions{
		LibPath:               cfg.OrtLibPath,
		DetInputSize:          cfg.DetInputSize,
		Workers:               cfg.Workers,
		DetThreshold:          float32(cfg.DetThreshold),
		NMSIoU:                float32(cfg.NMSIoU),
		DetectorInputName:     cfg.DetectorInputName,
		EmbedderInputName:     cfg.EmbedderInputName,
		EmbedderOutputName:    cfg.EmbedderOutputName,
		DetectorFormat:        cfg.DetectorFormat,
		LandmarkVarianceBaked: cfg.LandmarkVarianceBaked != nil && *cfg.LandmarkVarianceBaked,
		UseCoreML:             cfg.CoreML,
		ArcBatchSize:          cfg.ArcBatchSize,
	})
	if err != nil {
		logger.Error("onnx embedder init failed, falling back to stub", "err", err)
		return embedder.NewStub(), "stub"
	}
	return emb, "onnx"
}

// ensureModels auto-fetches missing ONNX weights using cfg's URLs
// + MinBytes heuristics. assetfetch.Ensure logs "asset already present"
// vs "asset fetched" itself; this function only logs missing-while-
// disabled diagnostics and propagates per-file fetch failures.
func ensureModels(ctx context.Context, cfg *config.Config, logger *slog.Logger) {
	type pair struct {
		kind, dst, url string
		min            int
	}
	pairs := []pair{
		{"detector", cfg.DetectorModelPath, cfg.DetectorModelURL, 1 * 1024 * 1024},
		{"recognizer", cfg.EmbedderModelPath, cfg.EmbedderModelURL, 5 * 1024 * 1024},
	}
	for _, p := range pairs {
		if p.dst == "" {
			continue
		}
		if !cfg.AutoDownload {
			if _, err := os.Stat(p.dst); errors.Is(err, os.ErrNotExist) {
				logger.Warn("model missing and auto-download disabled", "kind", p.kind, "path", p.dst)
			}
			continue
		}
		if p.url == "" {
			if _, err := os.Stat(p.dst); errors.Is(err, os.ErrNotExist) {
				logger.Warn("model missing and no URL configured", "kind", p.kind, "path", p.dst)
			}
			continue
		}
		if _, err := assetfetch.Ensure(ctx, p.dst, p.url, assetfetch.Options{
			MinBytes: p.min,
			Magic:    []byte{0x08, 0x06}, // ONNX ir_version varint tag (0x08) + protobuf wire type LENGTH_DELIMITED (0x06)
			Logger:   logger,
		}); err != nil {
			logger.Error("model download failed; embedder may fall back to stub", "kind", p.kind, "dst", p.dst, "err", err)
		}
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "regognition: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// Config first so the logger honors cfg.LogLevel + cfg.Color on
	// every subsequent event. flag.Parse / config errors before this
	// point print via fmt.Fprintf (only window when log lines bypass
	// prettylog).
	cfg, err := config.Load(flag.CommandLine, os.Args[1:])
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		flag.Usage()
		return err
	}
	// --debug / [ui] debug = true forces LogLevel to "debug" so the
	// per-face DEBUG lines (pre-align scores, decode stats, etc.)
	// become visible. We do this BEFORE creating the logger so the
	// override is honoured from the first emitted line.
	if cfg.Debug {
		cfg.LogLevel = "debug"
	}
	logger := slog.New(prettylog.New(os.Stderr, prettylog.ParseLevel(cfg.LogLevel), cfg.Color))
	slog.SetDefault(logger)
	logger.Info("config loaded", "summary", cfg.Summary())

	// Set up the signal handler BEFORE the heavy I/O below —
	// ensureModels + buildEmbedder can take several seconds together
	// (ORT initialisation, model load). Without this, a Ctrl-C
	// arriving during that window has no Go handler registered yet
	// and the OS default action (terminate) fires silently; the
	// operator never sees an "interrupt received" log.
	ctx, cancel := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
	defer cancel()
	// Dedicated signal channel for the watchdog log. We do NOT use
	// `select { case <-sigCh: case <-ctx.Done(): }` because when
	// SIGINT arrives BOTH channels become ready simultaneously
	// (signal.NotifyContext catches the signal AND cancels ctx),
	// and Go's select picks randomly — half the time the goroutine
	// would silently exit via ctx.Done() and the operator never
	// sees the warning. Plain receive on sigCh guarantees the log
	// fires on a real signal. The goroutine lives until the process
	// exits, which is fine for a one-shot CLI.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	// Unregister the OS signal handler on run() return. No-op for
	// this one-shot CLI (the process is about to exit), but textbook
	// for any future refactor that turns run() into a library/daemon
	// entry point where sigCh would otherwise stay registered for
	// the process lifetime.
	defer signal.Stop(sigCh)
	// Force-exit safety net. yalue/onnxruntime_go's Run() is a
	// blocking cgo C call that does NOT accept context, so once a
	// worker is in Run() the pipeline can't shut down until the C
	// call returns. Without a hard timeout, a Ctrl-C landing mid-Run
	// leaves the operator waiting indefinitely. 5s gives in-flight
	// C calls room to finish cleanly; on timeout we os.Exit(130)
	// (128 + SIGINT) so the process dies regardless of goroutine
	// state.
	//
	// The timer is owned by run() (not the goroutine) so the
	// defer below can actually stop it on a clean exit — a
	// defer inside a goroutine that blocks on <-sigCh forever
	// is dead code. atomic.Pointer is used because the watchdog
	// goroutine writes the timer on SIGINT while the defer reads
	// it on run() return; the two synchronization points are
	// independent (signal.Notify vs signal.NotifyContext) with no
	// shared happens-before chain, so a plain *time.Timer would
	// race per the Go memory model and the race detector would
	// flag it.
	var forceExit atomic.Pointer[time.Timer]
	defer func() {
		if t := forceExit.Load(); t != nil {
			t.Stop()
		}
	}()
	go func() {
		sig := <-sigCh
		logger.Warn("interrupt received, shutting down gracefully",
			"signal", sig.String())
		forceExit.Store(time.AfterFunc(5*time.Second, func() {
			logger.Warn("force exit after 5s shutdown grace period")
			os.Exit(130)
		}))
	}()

	// First-run model availability check. If cfg.DetectorModelPath /
	// cfg.EmbedderModelPath are missing AND cfg.AutoDownload is true AND
	// a URL is configured, assetfetch.Ensure downloads them atomically
	// (.tmp+rename, magic-byte check, progress logging). Failures are
	// non-fatal — buildEmbedder below still picks the stub if the files
	// are missing in the end. Pass ctx so Ctrl-C cancels any
	// in-flight model download.
	ensureModels(ctx, cfg, logger)

	// Embedder selection: ONNX-backed when libonnxruntime + both model
	// files are reachable; deterministic stub otherwise (logging the
	// fallback so it's never silent in production).
	inner, embKind := buildEmbedder(cfg, logger)
	var emb embedder.Embedder = inner
	logger.Info("embedder ready", "kind", embKind)

	// Optional SQLite embedding cache. When CachePath is set, every
	// ExtractFile path is wrapped so repeat runs over the same archive
	// skip RetinaFace+ArcFace for unchanged files.
	//
	// --debug bypasses the cache: the cache only stores embeddings, not
	// the 112×112 aligned face crops the FileDebugSink needs. A cache
	// hit would silently produce no debug output, which is more
	// confusing than the slowdown. main.go owns this decision so
	// cache.go stays a pure pass-through wrapper.
	var fc cache.FaceCache
	var bw *cache.BatchedWriter
	if cfg.CachePath != "" && !cfg.Debug {
		var err error
		fc, err = cache.Open(cfg.CachePath)
		if err != nil {
			return fmt.Errorf("cache open: %w", err)
		}
		emb = cache.NewCached(inner, fc, logger)
		// Batched writer: buffer cache writes and flush in batches
		// of 200 rows or every 1s, whichever comes first. This
		// eliminates per-photo SQLite write-lock contention on fresh
		// cache runs (8 workers no longer serialize on each INSERT).
		bw = cache.NewBatchedWriter(fc, logger, 200, 1*time.Second)
		logger.Info("embedding cache enabled", "path", cfg.CachePath)
	} else if cfg.CachePath != "" && cfg.Debug {
		logger.Warn("debug mode: bypassing embedding cache (debug crops require fresh inference)",
			"cache_path", cfg.CachePath)
	}

	// --debug: build a PersonaDebugSink that captures every detected
	// face from the persona selfies so the operator can see what the
	// detector saw on the reference images. Pre-creating the empty
	// debug/persona/ directory is critical: if the detector finds
	// 0 faces, the operator still needs the visible empty folder as
	// the explicit "detector saw nothing" signal. Without it they
	// would have to cross-reference log lines to figure out whether
	// the persona pass ran at all.
	var personaDebugSink embedder.DebugSink
	if cfg.Debug {
		personaDebugDir := filepath.Join(cfg.OutputDir, "debug", "persona")
		if err := os.MkdirAll(personaDebugDir, 0o755); err != nil {
			return fmt.Errorf("create persona debug dir: %w", err)
		}
		absPersonaDir, absPersonaErr := filepath.Abs(cfg.PersonaDir)
		if absPersonaErr != nil {
			return fmt.Errorf("resolve absolute persona dir: %w", absPersonaErr)
		}
		personaDebugSink = &pipeline.PersonaDebugSink{
			SourceDir: absPersonaDir,
			DebugDir:  filepath.Join(cfg.OutputDir, "debug"),
			Logger:    logger,
		}
		logger.Info("debug mode: persona crop sink ready", "dir", personaDebugDir)
	}

	// Persona: load selfies + extract embeddings.
	pers, err := persona.LoadWithLogger(ctx, cfg.PersonaDir, emb, personaDebugSink, logger)
	if err != nil {
		return fmt.Errorf("persona load: %w", err)
	}
	logger.Info("persona ready",
		"selfies", pers.SelfieCount(),
		"faces", pers.FaceCount(),
		"skipped", len(pers.Skipped),
		"dir", cfg.PersonaDir,
	)
	if len(pers.Skipped) > 0 {
		logger.Warn("persona: skipped selfies",
			"count", len(pers.Skipped),
			"paths", pers.Skipped,
		)
	}
	// Empty persona is a soft signal, not a fatal one: the Skipped
	// paths + the per-selfie Debug lines already point at the failing
	// detector input. Running the archive stage with len(p.Embeddings)==0
	// would just spend 25k ONNX inferences to produce 0 matches — let
	// the operator surface the symptom and choose the next step.
	if pers.FaceCount() == 0 {
		logger.Warn("no persona faces extracted; skipping archive stage",
			"selfies", pers.SelfieCount(),
			"skipped", len(pers.Skipped),
			"hint", "check that the embedder can detect faces on the selfies in "+cfg.PersonaDir,
		)
		return nil
	}

	// Resolve the search dir to an absolute path BEFORE building the
	// FileDebugSink and BEFORE handing it to pipeline.Run. The
	// pipeline normalises internally as well, but the sink needs the
	// abs path now (its OnFace runs before pipeline.Run enters its
	// abs step) to do filepath.Rel against the same form the pipeline
	// will use.
	absSearchDir, err := filepath.Abs(cfg.SearchDir)
	if err != nil {
		return fmt.Errorf("resolve absolute search dir: %w", err)
	}

	// --debug: build the FileDebugSink that the pipeline will hand
	// to ExtractWithDebug. The sink re-derives dist per face and
	// saves a 112×112 JPEG for each match under <OutputDir>/debug,
	// mirroring the search-dir subdir tree.
	var debugSink embedder.DebugSink
	if cfg.Debug {
		debugDir := filepath.Join(cfg.OutputDir, "debug")
		debugSink = &pipeline.FileDebugSink{
			SearchDir: absSearchDir,
			DebugDir:  debugDir,
			Threshold: float32(cfg.Threshold),
			Persona:   pers.Embeddings,
			Logger:    logger,
		}
		logger.Info("debug mode: face crops will be saved", "dir", debugDir)
	}

	// Pipeline.
	stats := &pipeline.Stats{}
	if err := pipeline.Run(ctx, absSearchDir, pipeline.Options{
		Workers:           cfg.Workers,
		PrefetchBatchSize: 4,
		OutputDir:         cfg.OutputDir,
		TargetDimension:   cfg.TargetDimension,
		Threshold:         float32(cfg.Threshold),
		Embedder:          emb,
		Persona:           pers,
		Cache:             fc,
		Writer:            bw,
		DirSkip:           cfg.DirSkip,
		SkipFiles:         cfg.SkipFiles,
		Extensions:        cfg.Extensions,
		Logger:            logger,
		Stats:             stats,
		DebugSink:         debugSink,
	}); err != nil {
		return fmt.Errorf("pipeline: %w", err)
	}
	// emb.Close() runs in a goroutine with a 3s timeout. ORT's
	// DestroyEnvironment can stall if any session is mid-Run, and we
	// don't want a stuck teardown to defeat the Ctrl-C the operator
	// just pressed. The 3s budget is generous for a clean shutdown
	// of N pooled sessions; on timeout we log and exit anyway.
	defer func() {
		// Flush any remaining batched cache writes before closing.
		if bw != nil {
			bw.Close()
		}
		closeDone := make(chan struct{})
		go func() {
			defer close(closeDone)
			if err := emb.Close(); err != nil {
				logger.Warn("embedder close error", "err", err)
			}
		}()
		select {
		case <-closeDone:
		case <-time.After(3 * time.Second):
			logger.Warn("embedder close timeout, exiting anyway")
		}
	}()
	return nil
}

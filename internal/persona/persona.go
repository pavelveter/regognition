// Package persona loads one or more selfie images of the target person
// from a directory and yields a flat list of face embeddings.
//
// The embeddings are produced by an embedder.Embedder, so the rest of the
// pipeline doesn't need to know about RetinaFace/ArcFace internals.
//
// If Load returns a Persona with FaceCount()==0 the caller is responsible
// for deciding what to do (typical CLI: warn and exit 0 without running
// the archive stage). Historically this case returned an error and the
// process exited non-zero, which made it hard to diagnose the upstream
// detector failure from the final pipe; the early-warning-and-skip path
// keeps the operator's terminal exitcode clean.
package persona

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/disintegration/imaging"

	"regognition/internal/embedder"
	"regognition/internal/scanner"
)

// Persona holds the target person's face embeddings.
//
// Embeddings is the FLATTENED list of all faces detected across all
// selfies: with N selfies and F_i faces on selfie i, Embeddings length
// equals sum(F_i). The pipeline matches an archive photo against the
// entire flat list (any selfie → any face → best distance wins).
type Persona struct {
	SourceDir string
	// Selfies are the candidates that were decoded without I/O error
	// (and for which ExtractFile at least returned without an error).
	Selfies []string
	// Skipped holds paths whose ExtractFile returned an error. The caller
	// can surface this so the user knows which selfies were dropped.
	Skipped []string
	// Embeddings is the FLATTENED list of all faces detected across all
	// selfies: with N selfies and F_i faces on selfie i, length equals
	// sum(F_i). The first N elements are zero values if no faces were
	// found (Load fails early in that case).
	Embeddings [][]float32
}

// Load walks dir, decodes each image, and extracts face embeddings.
//
// Individual selfie decode/extraction failures are skipped (logged via
// slog.Default at Debug) but the overall call fails if NO embeddings
// are produced.
//
// sink may be nil. When non-nil, the embedder is asked for a
// per-face DebugSink emission via ExtractWithDebug so the operator
// can see what the detector saw (or, when 0 faces detected,
// see an empty debug/persona/ folder as the explicit "detector saw
// nothing" signal). The sink fires per face per selfie but does NOT
// affect the returned embeddings — debug is observability-only.
func Load(ctx context.Context, dir string, emb embedder.Embedder, sink embedder.DebugSink) (*Persona, error) {
	if dir == "" {
		return nil, errors.New("persona: empty directory")
	}
	if emb == nil {
		return nil, errors.New("persona: nil embedder")
	}
	paths, err := scanner.Walk(dir)
	if err != nil {
		return nil, fmt.Errorf("persona: walk %q: %w", dir, err)
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("persona: no selfie images in %q", dir)
	}

	p := &Persona{SourceDir: dir, Selfies: make([]string, 0, len(paths))}
	for _, sp := range paths {
		// Allow Ctrl-C to break out of processing large directories.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// When a sink is supplied we need the decoded image so we
		// can call ExtractWithDebug (in-memory). ExtractFile would
		// also decode internally but doesn't take a sink, so the
		// alternate branch keeps debug observability opt-in.
		var faces [][]float32
		var extractErr error
		if sink != nil {
			img, openErr := imaging.Open(sp, imaging.AutoOrientation(true))
			if openErr != nil {
				slog.Default().Debug("selfie open failed", "path", sp, "err", openErr)
				p.Skipped = append(p.Skipped, sp)
				continue
			}
			faces, extractErr = emb.ExtractWithDebug(ctx, img, sp, sink)
		} else {
			faces, extractErr = emb.ExtractFile(ctx, sp)
		}
		if extractErr != nil {
			// Per-selfie failure is non-fatal; pipeline-level zero-embeddings
			// check below catches systemic problems. The failed path is
			// recorded so the caller can warn the user.
			slog.Default().Debug("selfie extract failed", "path", sp, "err", extractErr)
			p.Skipped = append(p.Skipped, sp)
			continue
		}
		slog.Default().Debug("selfie extract done", "path", sp, "faces", len(faces))
		p.Selfies = append(p.Selfies, sp)
		p.Embeddings = append(p.Embeddings, faces...)
	}

	if len(p.Embeddings) == 0 {
		// Don't fail: the empty-persona case is operator-actionable
		// (look at the "selfie extract failed" Debug lines + Skipped
		// paths surfaced at INFO), not a panic. Returning a valid
		// (empty) Persona lets the caller log a clear warning and
		// gracefully no-op the archive stage instead of dying with
		// regognition: exit status 1 and a "persona load: ..." chain
		// that's harder to triage.
		slog.Default().Warn("persona: extracted 0 face embeddings",
			"selfies", len(paths),
			"dir", dir,
			"skipped", len(p.Skipped),
		)
	}
	return p, nil
}

// SelfieCount returns how many selfie files contributed to this persona,
// regardless of how many faces were found in each.
func (p *Persona) SelfieCount() int { return len(p.Selfies) }

// FaceCount returns the total number of face vectors across all selfies.
func (p *Persona) FaceCount() int { return len(p.Embeddings) }

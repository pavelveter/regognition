// Package assetfetch downloads a binary asset over HTTPS with periodic
// progress logging, an atomic .tmp -> rename finalization, and a
// head-byte magic check. It is designed for the first-run auto-fetch
// of ONNX model weights where a partial failure must NEVER leave a
// corrupt file in the destination path.
//
// Lifecycle:
//  1. If dst already exists, size >= MinBytes and head matches Magic,
//     Ensure returns (false, nil) without further I/O.
//  2. Otherwise, removes any leftover dst+".tmp" (previous crashed run),
//     streams the body to dst+".tmp" while periodically emitting slog
//     progress lines.
//  3. Verifies the freshly downloaded file: MinBytes and (optionally)
//     Magic head bytes.
//  4. Atomic os.Rename to dst. On any failure along the way the tmp
//     file is removed so re-running Ensure can start clean.
//
// Network errors on the primary URL trigger failover to Mirrors in
// order. All mirrors failing returns a wrapped error.
package assetfetch

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Options configures Ensure. Zero values are filled with safe defaults.
//
// MinBytes and Magic are soft sanity checks used during validation of
// both existing and freshly downloaded files; they are independent
// (size-only, magic-only, or both are all valid).
type Options struct {
	// MinBytes is the minimum acceptable file size for existing and
	// newly downloaded files. Zero disables the size check.
	MinBytes int

	// Magic, if non-empty, is the expected head prefix of the file
	// (existing & newly downloaded). Zero disables the magic check.
	Magic []byte

	// Mirrors are tried in order after primary fails on transport
	// errors or non-2xx responses. ctx cancellation cuts the loop short.
	Mirrors []string

	// Logger receives progress + result events. nil → slog.Default().
	Logger *slog.Logger

	// Timeout is the per-URL network timeout. 0 → 60s.
	Timeout time.Duration

	// ProgressEvery is the period between progress log lines. 0 → 2s.
	// No line is emitted if the download finishes faster.
	ProgressEvery time.Duration

	// HTTPClient is the underlying http.Client. nil → http.DefaultClient.
	HTTPClient *http.Client
}

// Ensure guarantees dst exists and is valid (size + magic), downloading
// from primary (or a mirror) when it is missing or invalid. See
// package docs for full lifecycle.
//
// Returns downloaded=true on a successful fresh fetch; false when
// dst already satisfies MinBytes + Magic. Errors are wrapped with the
// last transport failure; tmp files are cleaned on every error path.
func Ensure(ctx context.Context, dst, primary string, opts Options) (bool, error) {
	if dst == "" {
		return false, errors.New("assetfetch: dst path is empty")
	}
	if primary == "" {
		return false, errors.New("assetfetch: primary URL is empty")
	}
	opts = fillOpts(opts)

	if ok, err := alreadyValid(dst, opts); err != nil {
		return false, err
	} else if ok {
		opts.Logger.Info("asset already present", "path", dst)
		return false, nil
	}

	urls := append([]string{primary}, opts.Mirrors...)
	var lastErr error
	for _, u := range urls {
		if u == "" {
			continue
		}
		if err := ctx.Err(); err != nil {
			return false, err
		}
		err := fetchOne(ctx, dst, u, opts)
		if err == nil {
			return true, nil
		}
		lastErr = err
		opts.Logger.Warn("asset fetch failed, trying next mirror",
			"url", u, "err", err)
	}

	return false, fmt.Errorf(
		"assetfetch: all %d mirror(s) failed for %q; last: %w",
		len(urls), filepath.Base(dst), lastErr)
}

// fillOpts drops in defaults for any nil/zero Options fields.
func fillOpts(o Options) Options {
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	if o.Timeout == 0 {
		o.Timeout = 60 * time.Second
	}
	if o.ProgressEvery == 0 {
		o.ProgressEvery = 2 * time.Second
	}
	if o.HTTPClient == nil {
		o.HTTPClient = http.DefaultClient
	}
	return o
}

// alreadyValid reports whether dst satisfies both MinBytes and Magic.
// It is a pure predicate — it never mutates filesystem state.
func alreadyValid(dst string, opts Options) (bool, error) {
	info, err := os.Stat(dst)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("assetfetch: stat %q: %w", dst, err)
	}
	if !info.Mode().IsRegular() {
		return false, fmt.Errorf("assetfetch: %q exists but is not a regular file", dst)
	}
	if opts.MinBytes > 0 && info.Size() < int64(opts.MinBytes) {
		opts.Logger.Info("re-fetching: existing file too small",
			"path", dst, "size", info.Size(), "min", opts.MinBytes)
		return false, nil
	}
	if len(opts.Magic) > 0 {
		f, err := os.Open(dst)
		if err != nil {
			return false, err
		}
		head := make([]byte, len(opts.Magic))
		if _, err := io.ReadFull(f, head); err != nil {
			_ = f.Close()
			opts.Logger.Info("re-fetching: cannot read existing head bytes",
				"path", dst, "err", err)
			return false, nil
		}
		_ = f.Close()
		if !bytes.Equal(head, opts.Magic) {
			opts.Logger.Info("re-fetching: existing file fails magic check",
				"path", dst, "want", fmt.Sprintf("%x", opts.Magic),
				"got", fmt.Sprintf("%x", head))
			return false, nil
		}
	}
	return true, nil
}

// fetchOne downloads url into dst via atomic .tmp + rename.
func fetchOne(parent context.Context, dst, url string, opts Options) error {
	ctx, cancel := context.WithTimeout(parent, opts.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "regognition/0 (assetfetch)")

	resp, err := opts.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("http do %q: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("status %d for %q: %s",
			resp.StatusCode, url, strings.TrimSpace(string(snippet)))
	}

	if dir := filepath.Dir(dst); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir %q: %w", dir, err)
		}
	}

	tmp := dst + ".tmp"
	_ = os.Remove(tmp) // clear leftover from a crashed previous run

	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create tmp %q: %w", tmp, err)
	}
	// Safety net: if a panic or unhandled error path exits before
	// the atomic rename, remove the orphaned .tmp file.
	defer func() {
		// os.Remove on a non-existent path is a no-op (returns nil).
		_ = os.Remove(tmp)
	}()

	tracker := newProgressTracker(opts.Logger, filepath.Base(dst),
		resp.ContentLength, opts.ProgressEvery)

	n, copyErr := io.Copy(io.MultiWriter(f, tracker.pipe()), resp.Body)
	tracker.stop()
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("copy %q -> %q: %w", url, tmp, copyErr)
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("close tmp %q: %w", tmp, closeErr)
	}

	if opts.MinBytes > 0 && n < int64(opts.MinBytes) {
		_ = os.Remove(tmp)
		return fmt.Errorf(
			"download too small: %d bytes from %q (min %d)",
			n, url, opts.MinBytes)
	}
	if len(opts.Magic) > 0 {
		if err := verifyMagic(tmp, opts.Magic); err != nil {
			_ = os.Remove(tmp)
			return fmt.Errorf("magic check on %q: %w", tmp, err)
		}
	}

	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename %q -> %q: %w", tmp, dst, err)
	}

	opts.Logger.Info("asset fetched",
		"path", dst, "url", url, "bytes", n)
	return nil
}

// verifyMagic reads len(magic) bytes from path and returns nil iff they
// equal magic.
func verifyMagic(path string, magic []byte) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	head := make([]byte, len(magic))
	if _, err := io.ReadFull(f, head); err != nil {
		return err
	}
	if !bytes.Equal(head, magic) {
		return fmt.Errorf(
			"expected magic %x, got %x", magic, head)
	}
	return nil
}

// progressTracker logs periodic download progress via a background
// goroutine that posts at opts.ProgressEvery intervals. writerPipe
// feeds the byte counter; calls to stop() terminate the goroutine and
// Wait until it has exited.
type progressTracker struct {
	logger  *slog.Logger
	asset   string
	total   int64
	written atomic.Int64

	writerPipe io.Writer // counts bytes; never blocks
	stopCh     chan struct{}
	wg         sync.WaitGroup
}

func newProgressTracker(logger *slog.Logger, asset string, total int64, every time.Duration) *progressTracker {
	pt := &progressTracker{
		logger: logger,
		asset:  asset,
		total:  total,
		stopCh: make(chan struct{}),
	}
	pt.writerPipe = pt // self-ref satisfies io.Writer through Write below
	pt.wg.Add(1)
	go pt.loop(every)
	return pt
}

// pipe returns the io.Writer half of the tracker (counts bytes).
func (p *progressTracker) pipe() io.Writer { return p.writerPipe }

// Write counts bytes; never returns an error to keep io.Copy honest.
func (p *progressTracker) Write(b []byte) (int, error) {
	p.written.Add(int64(len(b)))
	return len(b), nil
}

func (p *progressTracker) stop() {
	close(p.stopCh)
	p.wg.Wait()
}

func (p *progressTracker) loop(every time.Duration) {
	defer p.wg.Done()
	ticker := time.NewTicker(every)
	defer ticker.Stop()

	for {
		select {
		case <-p.stopCh:
			return
		case <-ticker.C:
			written := p.written.Load()
			if p.total > 0 {
				pct := float64(written) * 100.0 / float64(p.total)
				p.logger.Info("downloading",
					"asset", p.asset,
					"bytes", written,
					"total", p.total,
					"pct", fmt.Sprintf("%.1f%%", pct))
			} else {
				p.logger.Info("downloading",
					"asset", p.asset, "bytes", written)
			}
		}
	}
}

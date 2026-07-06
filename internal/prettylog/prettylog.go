// Package prettylog implements a slog.Handler that emits colorised,
// single-line records for interactive terminals. The output looks like:
//
//	14:22:01 ✓ INFO  config loaded       persona="persona"  workers=4
//	14:22:02 ⚠ WARN  model missing       kind="detector"  path="./models/..."
//	14:22:03 ✗ ERROR copy failed          src="a.jpg"      err="..."
//	14:22:04 … DEBUG extract done         path="x.jpg"     faces=2
//
// Color behaviour is governed by ColorMode at New() time:
//
//   - "always": force ANSI codes regardless of writer.
//   - "never":  never emit ANSI codes.
//   - "auto":   emit codes iff (a) $NO_COLOR is unset
//     (https://no-color.org/) AND (b) the writer is a *os.File
//     whose fd resolves to a TTY via golang.org/x/term.
//
// The handler is safe for concurrent use; one Mutex around Handle()
// serialises writes so record lines never interleave on a busy CLI.
package prettylog

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"

	"golang.org/x/term"
)

// ANSI escape codes — kept inline so the package has no extra
// non-stdlib deps beyond golang.org/x/term (TTY detection).
const (
	ansiReset   = "\x1b[0m"
	ansiBold    = "\x1b[1m"
	ansiDim     = "\x1b[2m"
	ansiRed     = "\x1b[31m"
	ansiGreen   = "\x1b[32m"
	ansiYellow  = "\x1b[33m"
	ansiMagenta = "\x1b[35m"
	ansiCyan    = "\x1b[36m"
	ansiGray    = "\x1b[90m"
)

const timeFormat = "15:04:05"

// levelStyle associates a level with the printed icon + color.
type levelStyle struct {
	icon  string
	color string
}

var levelStyles = map[slog.Level]levelStyle{
	slog.LevelDebug: {"…", ansiGray},
	slog.LevelInfo:  {"✓", ansiGreen},
	slog.LevelWarn:  {"⚠", ansiYellow},
	slog.LevelError: {"✗", ansiRed},
}

// Handler is the slog.Handler implementation.
type Handler struct {
	mu    sync.Mutex
	w     io.Writer
	level slog.Level
	color bool

	// Bound state from WithAttrs / WithGroup. Pre-bound attrs are
	// emitted BEFORE the record's own attrs on every Handle call.
	boundAttrs []slog.Attr
	groupPath  string
}

// New constructs a Handler. colorMode is one of "always", "never",
// "auto" (case-insensitive). Empty string behaves like "auto".
func New(w io.Writer, level slog.Level, colorMode string) *Handler {
	return &Handler{
		w:     w,
		level: level,
		color: shouldColor(w, colorMode),
	}
}

// shouldColor decides whether ANSI codes should be emitted. Honors
// $NO_COLOR (https://no-color.org/) before the TTY probe so CI logs
// stay plain even if a controlling terminal exists.
func shouldColor(w io.Writer, mode string) bool {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "always":
		return true
	case "never":
		return false
	default:
		if os.Getenv("NO_COLOR") != "" {
			return false
		}
		f, ok := w.(*os.File)
		if !ok {
			return false
		}
		return term.IsTerminal(int(f.Fd()))
	}
}

// Enabled reports whether the handler wants records at l.
func (h *Handler) Enabled(_ context.Context, l slog.Level) bool {
	return l >= h.level
}

// Handle emits a single record.
func (h *Handler) Handle(_ context.Context, r slog.Record) error {
	style, ok := levelStyles[r.Level]
	if !ok {
		style = levelStyle{"•", ansiMagenta}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	var b strings.Builder

	// Time prefix (dimmed when colored).
	if h.color {
		b.WriteString(ansiDim)
		b.WriteString(r.Time.Format(timeFormat))
		b.WriteString(ansiReset)
	} else {
		b.WriteString(r.Time.Format(timeFormat))
	}
	b.WriteByte(' ')

	// Icon + level badge.
	if h.color {
		fmt.Fprintf(&b, "%s%1s%s ", style.color, style.icon, ansiReset)
		fmt.Fprintf(&b, "%s%s%-5s%s ", ansiBold, style.color, r.Level.String(), ansiReset)
	} else {
		fmt.Fprintf(&b, "%s %-5s ", style.icon, r.Level.String())
	}

	// Message body.
	b.WriteString(r.Message)

	// Pre-bound attrs (those attached via WithAttrs / WithGroup).
	for _, a := range h.boundAttrs {
		writeKV(&b, qualifyKey(a.Key, h.groupPath), a.Value, h.color)
	}

	// Record-specific attrs.
	r.Attrs(func(a slog.Attr) bool {
		writeKV(&b, qualifyKey(a.Key, h.groupPath), a.Value, h.color)
		return true
	})

	b.WriteByte('\n')
	_, err := io.WriteString(h.w, b.String())
	return err
}

// WithAttrs returns a Handler that knows about the given attrs.
func (h *Handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	merged := make([]slog.Attr, 0, len(h.boundAttrs)+len(attrs))
	merged = append(merged, h.boundAttrs...)
	merged = append(merged, attrs...)
	return &Handler{
		w:          h.w,
		level:      h.level,
		color:      h.color,
		boundAttrs: merged,
		groupPath:  h.groupPath,
	}
}

// WithGroup returns a Handler that prefixes attr keys with the given
// (possibly nested) group path.
func (h *Handler) WithGroup(name string) slog.Handler {
	newPath := name
	if h.groupPath != "" {
		newPath = h.groupPath + "." + name
	}
	return &Handler{
		w:          h.w,
		level:      h.level,
		color:      h.color,
		boundAttrs: h.boundAttrs,
		groupPath:  newPath,
	}
}

// qualifyKey prefixes key with prefix when prefix is non-empty.
func qualifyKey(key, prefix string) string {
	if prefix == "" {
		return key
	}
	return prefix + "." + key
}

// writeKV emits one `  key=value` pair to b.
// slog.Value.String() includes quotes for strings — we strip them
// because the pretty format has its own color-driven framing.
func writeKV(b *strings.Builder, key string, v slog.Value, color bool) {
	s := v.String()
	if v.Kind() == slog.KindString {
		if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
			s = s[1 : len(s)-1]
		}
	}
	if color {
		b.WriteString("  ")
		b.WriteString(ansiDim)
		b.WriteString(key)
		b.WriteString(ansiReset)
		b.WriteByte('=')
		b.WriteString(ansiCyan)
		b.WriteString(s)
		b.WriteString(ansiReset)
	} else {
		fmt.Fprintf(b, "  %s=%s", key, s)
	}
}

// ParseLevel maps a lenient level string ("debug", "info", "warn",
// "warning", "error") to a slog.Level. Unknown values fall back to
// slog.LevelInfo so a typo silently keeps the program readable.
func ParseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error", "err":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

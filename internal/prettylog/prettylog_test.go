package prettylog

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestPretty_NoColor_NoANSI(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelDebug, "never")
	logger := slog.New(h)
	logger.Info("hello", "key", "value", "num", 42)
	out := buf.String()
	for _, want := range []string{"hello", "INFO", "key=value", "num=42"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in %q", want, out)
		}
	}
	if strings.Contains(out, "\x1b[") {
		t.Fatalf("ANSI codes leaked in 'never' mode: %q", out)
	}
}

func TestPretty_Always_EmitsANSI(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelDebug, "always")
	logger := slog.New(h)
	logger.Info("hello")
	if !strings.Contains(buf.String(), "\x1b[") {
		t.Fatalf("missing ANSI codes in 'always' mode: %q", buf.String())
	}
}

func TestPretty_LevelFilter_SuppressesBelow(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelWarn, "never")
	logger := slog.New(h)
	logger.Info("info-suppressed")
	logger.Warn("warn-emitted")
	out := buf.String()
	if strings.Contains(out, "info-suppressed") {
		t.Fatalf("Info should be filtered: %q", out)
	}
	if !strings.Contains(out, "warn-emitted") {
		t.Fatalf("Warn should pass: %q", out)
	}
}

func TestPretty_WithAttrs_BoundAttrEmitted(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelDebug, "never")
	logger := slog.New(h).With("session", "abc")
	logger.Info("hi")
	out := buf.String()
	if !strings.Contains(out, "session=abc") {
		t.Fatalf("missing bound attr: %q", out)
	}
}

func TestPretty_WithGroup_PrefixedKey(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelDebug, "never")
	logger := slog.New(h).WithGroup("pipeline")
	logger.Info("done", "elapsed", "5s")
	out := buf.String()
	if !strings.Contains(out, "pipeline.elapsed=5s") {
		t.Fatalf("missing grouped key: %q", out)
	}
}

func TestPretty_AllLevelsRendered(t *testing.T) {
	var buf bytes.Buffer
	h := New(&buf, slog.LevelDebug, "never")
	logger := slog.New(h)
	logger.Debug("d")
	logger.Info("i")
	logger.Warn("w")
	logger.Error("e")
	out := buf.String()
	for _, lvl := range []string{"DEBUG", "INFO", "WARN", "ERROR"} {
		if !strings.Contains(out, lvl) {
			t.Fatalf("missing %s in output: %q", lvl, out)
		}
	}
}

func TestParseLevel_AcceptsSynonyms(t *testing.T) {
	cases := map[string]slog.Level{
		"debug":   slog.LevelDebug,
		"info":    slog.LevelInfo,
		"warn":    slog.LevelWarn,
		"warning": slog.LevelWarn,
		"error":   slog.LevelError,
		"err":     slog.LevelError,
		"":        slog.LevelInfo, // unknown / empty defaults to info
		"bogus":   slog.LevelInfo,
		"  WARN ": slog.LevelWarn,
	}
	for in, want := range cases {
		if got := ParseLevel(in); got != want {
			t.Errorf("ParseLevel(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestShouldColor_RespectsNoColorEnv(t *testing.T) {
	t.Setenv("NO_COLOR", "1")
	if shouldColor(new(bytes.Buffer), "auto") {
		t.Fatalf("NO_COLOR should force plain output")
	}
}

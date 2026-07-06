// codec_test.go — round-trip tests for the binary codec + HashFile
// sanity check. Pure Go, no SQLite or ONNX required.

package cache

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEncodeDecodeRoundtrip(t *testing.T) {
	faces := [][]float32{
		make([]float32, EmbedDim),
		make([]float32, EmbedDim),
	}
	for i := range faces[0] {
		faces[0][i] = float32(i) * 0.01
		faces[1][i] = float32(EmbedDim-i) * -0.001
	}
	blob := encodeEmbeddings(faces)
	got, err := decodeEmbeddings(blob, len(faces))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(faces) {
		t.Fatalf("face count: got %d want %d", len(got), len(faces))
	}
	for i := range faces {
		for j := range faces[i] {
			if got[i][j] != faces[i][j] {
				t.Fatalf("face %d idx %d: got %v want %v", i, j, got[i][j], faces[i][j])
			}
		}
	}
}

func TestEncodeDecodeEmpty(t *testing.T) {
	blob := encodeEmbeddings(nil)
	if len(blob) != 0 {
		t.Fatalf("empty blob expected, got %d bytes", len(blob))
	}
	got, err := decodeEmbeddings(blob, 0)
	if err != nil {
		t.Fatalf("decode empty: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected 0 faces, got %d", len(got))
	}
}

func TestDecodeCorrupt(t *testing.T) {
	// Wrong length blob for the claimed face count must error out.
	faces := [][]float32{make([]float32, EmbedDim)}
	blob := encodeEmbeddings(faces)
	if _, err := decodeEmbeddings(blob[:len(blob)-4], len(faces)); err == nil {
		t.Fatal("expected decode error on truncated blob")
	}
}

func TestHashFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	h1, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	h2, _ := HashFile(p)
	if h1 != h2 {
		t.Fatalf("hash unstable: %s vs %s", h1, h2)
	}
	// Modify file → hash must change.
	if err := os.WriteFile(p, []byte("hello!"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Allow filesystem mtime to advance.
	stat, _ := os.Stat(p)
	_ = stat.ModTime()
	h3, _ := HashFile(p)
	if h1 == h3 {
		t.Fatalf("hash unchanged after content rewrite: %s", h1)
	}
}

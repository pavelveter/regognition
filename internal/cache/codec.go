// codec.go — encoding/decoding of embedding vectors as a binary
// stream, and the file-hash used as the cache freshness field.

package cache

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
)

// EmbedDim is the cache's expected per-face vector length.
// RetinaFace/ArcFace (MobileFaceNet, ResNet-50) all emit 512-d
// vectors; if you onboard a model with a different dim, update this
// constant AND the encoding/decoding call sites.
const EmbedDim = 512

// HashFile returns the SHA1 of "path|mtime-nano|size" — the same
// formula called out in TODO.md §3. Identical inputs (same path,
// same content) produce identical hashes; modify either content
// (size) or metadata (mtime) and the hash changes, marking the
// cached entry stale.
func HashFile(path string) (string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	h := sha1.New()
	if _, err := h.Write([]byte(path)); err != nil {
		// sha1.Writer.Write never returns a non-nil error in Go 1.x.
		return "", fmt.Errorf("cache: hash path: %w", err)
	}
	if err := binary.Write(h, binary.LittleEndian, fi.Size()); err != nil {
		return "", fmt.Errorf("cache: hash size: %w", err)
	}
	if err := binary.Write(h, binary.LittleEndian, fi.ModTime().UnixNano()); err != nil {
		return "", fmt.Errorf("cache: hash mtime: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// encodeEmbeddings packs a sequence of face vectors back-to-back as
// float32 little-endian. Total bytes = len(faces) * EmbedDim * 4.
// An empty input produces an empty ([]byte)(nil), which SQLite stores
// as a zero-length BLOB.
func encodeEmbeddings(faces [][]float32) []byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(faces)*EmbedDim*4))
	for _, v := range faces {
		if len(v) != EmbedDim {
			// Panic is appropriate here: this indicates a programming
			// error (model output mismatch), not a data error.
			panic(fmt.Sprintf("cache: encodeEmbeddings: vector dim %d != EmbedDim %d", len(v), EmbedDim))
		}
		for _, x := range v {
			_ = binary.Write(buf, binary.LittleEndian, x)
		}
	}
	return buf.Bytes()
}

// decodeEmbeddings reverses encodeEmbeddings. count is the number of
// face vectors stored; the blob length is validated against
// len(faces) * EmbedDim * 4 to catch corrupt rows early.
func decodeEmbeddings(data []byte, count int) ([][]float32, error) {
	if count < 0 {
		return nil, errors.New("cache: negative face count")
	}
	expected := count * EmbedDim * 4
	if count == 0 {
		if len(data) != 0 {
			return nil, fmt.Errorf("cache: 0 faces but blob has %d bytes", len(data))
		}
		return [][]float32{}, nil
	}
	if len(data) != expected {
		return nil, fmt.Errorf("cache: blob length %d does not match %d faces × %d floats × 4 bytes (= %d)",
			len(data), count, EmbedDim, expected)
	}
	faces := make([][]float32, count)
	rdr := bytes.NewReader(data)
	for i := range faces {
		v := make([]float32, EmbedDim)
		if err := binary.Read(rdr, binary.LittleEndian, &v); err != nil {
			return nil, fmt.Errorf("cache: read face %d: %w", i, err)
		}
		faces[i] = v
	}
	return faces, nil
}

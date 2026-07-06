// stub.go — deterministic non-ML embedder used as a placeholder until a
// real RetinaFace + ArcFace ONNX implementation lands.
//
// Strategy:
//  - Take the center pixel's R, G, B as the first three components.
//  - Fill the remaining 509 components with a deterministic LCG seeded
//    from a hash of (path, sampled pixels). Identical inputs -> identical
//    vectors.
//
// End-to-end test recipe: drop the SAME image into both persona/ and
// dir_search/. The stub will produce distance 0.0 (best possible match).
// Different images produce non-trivial distances, which lets you sanity
// check the threshold behavior.
//
// Replace NewStub() with the ONNX implementation once libonnxruntime is
// installed; no other code must change.

package embedder

import (
	"context"
	"encoding/binary"
	"hash/fnv"
	"image"
	"math"

	"github.com/disintegration/imaging"
)

// Dim is the embedding dimensionality (matches ArcFace r50).
const Dim = 512

// Stub is the no-ML placeholder embedder.
type Stub struct{}

// NewStub returns the deterministic stub embedder.
func NewStub() *Stub { return &Stub{} }

func (Stub) Close() error { return nil }

// Extract returns a single 512-d vector for the input image.
//
// The stub intentionally returns one face per image regardless of content;
// a real implementation would run RetinaFace and return 0..N vectors.
func (Stub) Extract(_ context.Context, img image.Image) ([][]float32, error) {
	v := make([]float32, Dim)
	fillFromPixelAndHash(v, img, "")
	l2Normalize(v)
	return [][]float32{v}, nil
}

// ExtractWithDebug is the stub's debug variant. It returns the same
// single face as Extract and also notifies sink with a placeholder
// Aligned image: a CatmullRom downscale of the source to ArcFace's
// 112×112. The placeholder is NOT a real aligned face (the stub has
// no detector / 5-point regressor) — it's only useful for end-to-end
// plumbing tests where the operator wants to see "did the sink fire
// and did the file land where expected".
//
// sink may be nil, in which case the call is equivalent to Extract.
func (Stub) ExtractWithDebug(ctx context.Context, img image.Image, srcPath string, sink DebugSink) ([][]float32, error) {
	faces, err := (Stub{}).Extract(ctx, img)
	if err != nil {
		return nil, err
	}
	if sink != nil && len(faces) > 0 && img != nil {
		placeholder := imaging.Fit(img, ArcFaceInputSize, ArcFaceInputSize, imaging.CatmullRom)
		sink.OnFace(nil, srcPath, Face{
			Embedding: faces[0],
			BBox:      img.Bounds(),
			Aligned:   placeholder,
			Score:     0,
			Index:     0,
		})
	}
	return faces, nil
}

// ExtractFile reads the file via the imaging library (auto-orient, EXIF)
// then produces the deterministic vector.
func (Stub) ExtractFile(_ context.Context, path string) ([][]float32, error) {
	img, err := imaging.Open(path, imaging.AutoOrientation(true))
	if err != nil {
		return nil, err
	}
	v := make([]float32, Dim)
	fillFromPixelAndHash(v, img, path)
	l2Normalize(v)
	return [][]float32{v}, nil
}

// fillFromPixelAndHash seeds v[0..2] from the center pixel; the rest is
// filled from a deterministic LCG, seeded once via FNV over a quick subset
// of pixels and the path (if any).
func fillFromPixelAndHash(v []float32, img image.Image, salt string) {
	b := img.Bounds()
	cx, cy := b.Dx()/2, b.Dy()/2
	if cx < 0 {
		cx = 0
	}
	if cy < 0 {
		cy = 0
	}
	r, g, bl, _ := img.At(cx+b.Min.X, cy+b.Min.Y).RGBA()
	v[0] = float32(r) / 65535.0
	v[1] = float32(g) / 65535.0
	v[2] = float32(bl) / 65535.0

	h := fnv.New64a()
	if salt != "" {
		_, _ = h.Write([]byte(salt))
	}
	_ = binary.Write(h, binary.LittleEndian, int64(b.Dx()))
	_ = binary.Write(h, binary.LittleEndian, int64(b.Dy()))

	// Sample at most ~8K pixels so we stay fast on huge archives.
	step := b.Dx() / 64
	if step < 1 {
		step = 1
	}
	for y := b.Min.Y; y < b.Max.Y && h.Size() < 8192; y += step {
		for x := b.Min.X; x < b.Max.X && h.Size() < 8192; x += step {
			r, g, bl, _ := img.At(x, y).RGBA()
			_ = binary.Write(h, binary.LittleEndian, uint16(r))
			_ = binary.Write(h, binary.LittleEndian, uint16(g))
			_ = binary.Write(h, binary.LittleEndian, uint16(bl))
		}
	}
	// LCG (MMIX constants) to spread the 64-bit seed into 509 floats.
	seed := h.Sum64()
	for i := 3; i < Dim; i++ {
		seed = seed*6364136223846793005 + 1442695040888963407
		// 1<<31 is fine as float32 (representable, just imprecise); the
		// int32() cast would overflow because 2^31 > MaxInt32.
		v[i] = float32(int32(seed>>32)) / float32(1<<31)
	}
}

func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := 1.0 / math.Sqrt(sum)
	for i := range v {
		v[i] *= float32(inv)
	}
}

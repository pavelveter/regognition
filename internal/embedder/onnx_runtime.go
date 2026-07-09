// onnx_runtime.go — preprocessing, retinaface decode + NMS, and the
// 5-point affine alignment that produces the 112x112 crop the
// recognizer expects. Driven by the ONNXEmbedder in onnx.go.

package embedder

import (
	"errors"
	"fmt"
	"image"
	"log/slog"
	"math"
	"sort"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// resizePool reuses 640×640 RGBA buffers for detector preprocessing.
// Each buffer is ~1.6 MB; the pool prevents one allocation per image.
var resizePool = sync.Pool{
	New: func() any {
		return image.NewRGBA(image.Rect(0, 0, 640, 640))
	},
}

// --- Detector preprocessing ---

// minFacePx is the minimum bounding-box dimension (in original image
// pixels) below which a detection is considered a false positive.
// Typical faces in photos are ≥80px wide; the detector's weak
// texture-pattern false positives tend to be 30–50px.
const minFacePx = 50

// detectMean is the BGR mean subtraction used by the biubug6
// Pytorch_Retinaface MobileNet0.25 export (retinaface_mnet025_v2.onnx).
var detectMean = [3]float32{104.0, 117.0, 123.0}

// preprocessRetinaFace packs src into dst as CHW BGR float32 with
// per-channel mean subtraction. The image is resized to size×size
// (squashed, matching biubug6 training convention).
func preprocessRetinaFace(src image.Image, size int, dst *ort.Tensor[float32]) error {
	if dst == nil {
		return errors.New("preprocess: nil tensor")
	}
	if src.Bounds().Empty() {
		return errors.New("preprocess: empty image bounds")
	}
	// Get a pooled buffer for the resized image
	var resized *image.RGBA
	if size == 640 {
		resized = resizePool.Get().(*image.RGBA)
		defer resizePool.Put(resized)
	}
	resized = resizeBilinear(size, size, src, resized)
	b := resized.Bounds()
	if b.Dx() != size || b.Dy() != size {
		return errors.New("preprocess: resize produced unexpected shape")
	}
	tData := dst.GetData()
	if len(tData) != 3*size*size {
		return errors.New("preprocess: tensor buffer size mismatch")
	}
	// Direct Pix access — resizeBilinear always returns *image.RGBA
	pix := resized.Pix
	stride := size * size
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			off := (y*resized.Stride + x*4)
			b8 := float32(pix[off+2]) // B
			g8 := float32(pix[off+1]) // G
			r8 := float32(pix[off])   // R
			tData[y*size+x] = b8 - detectMean[0]
			tData[stride+y*size+x] = g8 - detectMean[1]
			tData[2*stride+y*size+x] = r8 - detectMean[2]
		}
	}
	return nil
}

// resizeBilinear is a minimal bilinear resampler; avoids pulling
// extra dependencies for one square-resize use site. If out is nil,
// a new image.RGBA is allocated; otherwise out is reused (must match
// dstW×dstH). Uses type-switch for direct Pix access when possible.
func resizeBilinear(dstW, dstH int, src image.Image, out *image.RGBA) *image.RGBA {
	srcBounds := src.Bounds()
	srcW, srcH := srcBounds.Dx(), srcBounds.Dy()
	if srcW == 0 || srcH == 0 || dstW == 0 || dstH == 0 {
		if out == nil {
			return image.NewRGBA(image.Rect(0, 0, dstW, dstH))
		}
		return out
	}
	if out == nil {
		out = image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	}
	scaleX := float64(srcW) / float64(dstW)
	scaleY := float64(srcH) / float64(dstH)

	// Type-switch for direct Pix access — avoids interface dispatch per pixel
	switch v := src.(type) {
	case *image.RGBA:
		resizeRGBADirect(v, out, srcBounds, scaleX, scaleY, dstW, dstH)
	case *image.NRGBA:
		resizeNRGBADirect(v, out, srcBounds, scaleX, scaleY, dstW, dstH)
	case *image.YCbCr:
		resizeYCbCrDirect(v, out, srcBounds, scaleX, scaleY, dstW, dstH)
	default:
		resizeGenericDirect(src, out, srcBounds, scaleX, scaleY, dstW, dstH)
	}
	return out
}

// resizeRGBADirect resizes with direct Pix access for *image.RGBA source.
func resizeRGBADirect(src *image.RGBA, out *image.RGBA, srcBounds image.Rectangle, scaleX, scaleY float64, dstW, dstH int) {
	srcPix := src.Pix
	srcStride := src.Stride
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < dstH; y++ {
		sy := (float64(y)+0.5)*scaleY - 0.5
		sy0 := clampInt(int(math.Floor(sy)), 0, srcBounds.Dy()-2)
		fy := sy - float64(sy0)
		for x := 0; x < dstW; x++ {
			sx := (float64(x)+0.5)*scaleX - 0.5
			sx0 := clampInt(int(math.Floor(sx)), 0, srcBounds.Dx()-2)
			fx := sx - float64(sx0)
			sy0a := srcBounds.Min.Y + sy0
			sx0a := srcBounds.Min.X + sx0
			off00 := sy0a*srcStride + sx0a*4
			off01 := off00 + 4
			off10 := off00 + srcStride
			off11 := off10 + 4
			// Pix stores 8-bit values; multiply by 257 to match RGBA() premultiplied 16-bit
			r := lerp(lerp(float64(srcPix[off00])*257, float64(srcPix[off01])*257, fx),
				lerp(float64(srcPix[off10])*257, float64(srcPix[off11])*257, fx), fy)
			g := lerp(lerp(float64(srcPix[off00+1])*257, float64(srcPix[off01+1])*257, fx),
				lerp(float64(srcPix[off10+1])*257, float64(srcPix[off11+1])*257, fx), fy)
			b := lerp(lerp(float64(srcPix[off00+2])*257, float64(srcPix[off01+2])*257, fx),
				lerp(float64(srcPix[off10+2])*257, float64(srcPix[off11+2])*257, fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(b) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// resizeNRGBADirect resizes with direct Pix access for *image.NRGBA source.
func resizeNRGBADirect(src *image.NRGBA, out *image.RGBA, srcBounds image.Rectangle, scaleX, scaleY float64, dstW, dstH int) {
	srcPix := src.Pix
	srcStride := src.Stride
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < dstH; y++ {
		sy := (float64(y)+0.5)*scaleY - 0.5
		sy0 := clampInt(int(math.Floor(sy)), 0, srcBounds.Dy()-2)
		fy := sy - float64(sy0)
		for x := 0; x < dstW; x++ {
			sx := (float64(x)+0.5)*scaleX - 0.5
			sx0 := clampInt(int(math.Floor(sx)), 0, srcBounds.Dx()-2)
			fx := sx - float64(sx0)
			sy0a := srcBounds.Min.Y + sy0
			sx0a := srcBounds.Min.X + sx0
			off00 := sy0a*srcStride + sx0a*4
			off01 := off00 + 4
			off10 := off00 + srcStride
			off11 := off10 + 4
			// Pix stores 8-bit values; multiply by 257 to match RGBA() premultiplied 16-bit
			r := lerp(lerp(float64(srcPix[off00])*257, float64(srcPix[off01])*257, fx),
				lerp(float64(srcPix[off10])*257, float64(srcPix[off11])*257, fx), fy)
			g := lerp(lerp(float64(srcPix[off00+1])*257, float64(srcPix[off01+1])*257, fx),
				lerp(float64(srcPix[off10+1])*257, float64(srcPix[off11+1])*257, fx), fy)
			b := lerp(lerp(float64(srcPix[off00+2])*257, float64(srcPix[off01+2])*257, fx),
				lerp(float64(srcPix[off10+2])*257, float64(srcPix[off11+2])*257, fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(b) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// resizeYCbCrDirect resizes with direct Pix access for *image.YCbCr source.
func resizeYCbCrDirect(src *image.YCbCr, out *image.RGBA, srcBounds image.Rectangle, scaleX, scaleY float64, dstW, dstH int) {
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < dstH; y++ {
		sy := (float64(y)+0.5)*scaleY - 0.5
		sy0 := clampInt(int(math.Floor(sy)), 0, srcBounds.Dy()-2)
		fy := sy - float64(sy0)
		for x := 0; x < dstW; x++ {
			sx := (float64(x)+0.5)*scaleX - 0.5
			sx0 := clampInt(int(math.Floor(sx)), 0, srcBounds.Dx()-2)
			fx := sx - float64(sx0)
			sy0a := srcBounds.Min.Y + sy0
			sx0a := srcBounds.Min.X + sx0
			r00, g00, b00, _ := src.At(sx0a, sy0a).RGBA()
			r01, g01, b01, _ := src.At(sx0a+1, sy0a).RGBA()
			r10, g10, b10, _ := src.At(sx0a, sy0a+1).RGBA()
			r11, g11, b11, _ := src.At(sx0a+1, sy0a+1).RGBA()
			r := lerp(lerp(float64(r00), float64(r01), fx), lerp(float64(r10), float64(r11), fx), fy)
			g := lerp(lerp(float64(g00), float64(g01), fx), lerp(float64(g10), float64(g11), fx), fy)
			b := lerp(lerp(float64(b00), float64(b01), fx), lerp(float64(b10), float64(b11), fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(b) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// resizeGenericDirect is the fallback for unknown image types.
func resizeGenericDirect(src image.Image, out *image.RGBA, srcBounds image.Rectangle, scaleX, scaleY float64, dstW, dstH int) {
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < dstH; y++ {
		sy := (float64(y)+0.5)*scaleY - 0.5
		sy0 := clampInt(int(math.Floor(sy)), 0, srcBounds.Dy()-2)
		fy := sy - float64(sy0)
		for x := 0; x < dstW; x++ {
			sx := (float64(x)+0.5)*scaleX - 0.5
			sx0 := clampInt(int(math.Floor(sx)), 0, srcBounds.Dx()-2)
			fx := sx - float64(sx0)
			sy0a := srcBounds.Min.Y + sy0
			sx0a := srcBounds.Min.X + sx0
			r00, g00, b00, _ := src.At(sx0a, sy0a).RGBA()
			r01, g01, b01, _ := src.At(sx0a+1, sy0a).RGBA()
			r10, g10, b10, _ := src.At(sx0a, sy0a+1).RGBA()
			r11, g11, b11, _ := src.At(sx0a+1, sy0a+1).RGBA()
			r := lerp(lerp(float64(r00), float64(r01), fx), lerp(float64(r10), float64(r11), fx), fy)
			g := lerp(lerp(float64(g00), float64(g01), fx), lerp(float64(g10), float64(g11), fx), fy)
			b := lerp(lerp(float64(b00), float64(b01), fx), lerp(float64(b10), float64(b11), fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(b) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func lerp(a, b, t float64) float64 { return a*(1-t) + b*t }

func clampUint16(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 65535 {
		return 65535
	}
	return v
}

// --- RetinaFace decoder ---

// retinaVariance matches the training-time variance used by the
// biubug6/Pytorch_Retinaface exporter.
var retinaVariance = [2]float32{0.1, 0.2}

// retinaMinSizes gives the two anchor sizes (in INPUT-PIXEL units,
// i.e. relative to det_input_size) generated at each stride, in the
// same order biubug6's prior_box.py uses.
var retinaMinSizes = map[int][2]float32{
	8:  {16, 32},
	16: {64, 128},
	32: {256, 512},
}

// prior is one anchor box in NORMALIZED [0,1] coordinates relative to
// the detector's square input (cx, cy, width, height).
type prior struct {
	cx, cy, sx, sy float32
}

var priorCache sync.Map // key: struct{stride,size} -> []prior

// priorsForStride returns the (cached) prior boxes for one stride at
// a given detector input size, generated in raster order (row-major
// cell, then anchor-within-cell) matching the ONNX export's per-anchor
// row ordering.
func priorsForStride(stride, detInputSize int) []prior {
	type key struct{ stride, size int }
	k := key{stride, detInputSize}
	if v, ok := priorCache.Load(k); ok {
		return v.([]prior)
	}
	sizes, ok := retinaMinSizes[stride]
	if !ok {
		return nil
	}
	h := detInputSize / stride
	out := make([]prior, 0, h*h*len(sizes))
	for ay := 0; ay < h; ay++ {
		for ax := 0; ax < h; ax++ {
			cx := (float32(ax) + 0.5) * float32(stride) / float32(detInputSize)
			cy := (float32(ay) + 0.5) * float32(stride) / float32(detInputSize)
			for _, s := range sizes {
				out = append(out, prior{
					cx: cx, cy: cy,
					sx: s / float32(detInputSize),
					sy: s / float32(detInputSize),
				})
			}
		}
	}
	priorCache.Store(k, out)
	return out
}

// face is one decoded face: bbox + 5 landmarks in ORIGINAL image
// coordinates plus confidence.
type face struct {
	X1, Y1, X2, Y2 float32
	Landmarks      [5][2]float32
	Score          float32
}

// decodeRetinaFaces consumes pre-grouped RetinaFace outputs (one
// entry per stride) and returns the NMS-filtered face set in ORIGINAL
// image coordinates. Uses anchor-based decoding with prior boxes and
// variance, matching the biubug6/Pytorch_Retinaface export format.
//
// landmarkVarianceBaked controls whether the ONNX export bakes
// variance[0] into landmark outputs. MobileNet does (true), ResNet50
// does not (false). When false, variance[0] is applied during decode.
func decodeRetinaFaces(groups []retinaOut, orig image.Rectangle, confThr, iouThr float32, landmarkVarianceBaked bool) []face {
	if len(groups) == 0 {
		return nil
	}
	var allFaces []face
	for _, g := range groups {
		if g.scores == nil || g.bboxes == nil || g.landmarks == nil || g.outputSize == 0 {
			continue
		}
		detSz := float32(detSizeFromGroup(g))
		priors := priorsForStride(g.stride, int(detSz))
		if len(priors) == 0 {
			continue
		}
		scaleX := float32(orig.Dx()) / detSz
		scaleY := float32(orig.Dy()) / detSz
		scoresData := g.scores.GetData()
		bboxesData := g.bboxes.GetData()
		lmData := g.landmarkssafe()
		var strideMax float32
		strideAbove := 0
		loggedTop := false
		for i, pr := range priors {
			prob := scoresData[i]
			if prob > strideMax {
				strideMax = prob
			}
			if prob >= confThr {
				strideAbove++
			}
			if prob < confThr {
				continue
			}
			bb := i * 4
			dx := bboxesData[bb+0]
			dy := bboxesData[bb+1]
			dw := bboxesData[bb+2]
			dh := bboxesData[bb+3]
			boxCx := pr.cx + dx*retinaVariance[0]*pr.sx
			boxCy := pr.cy + dy*retinaVariance[0]*pr.sy
			boxW := pr.sx * float32(math.Exp(float64(dw*retinaVariance[1])))
			boxH := pr.sy * float32(math.Exp(float64(dh*retinaVariance[1])))
			x1 := (boxCx - boxW/2) * detSz * scaleX
			y1 := (boxCy - boxH/2) * detSz * scaleY
			x2 := (boxCx + boxW/2) * detSz * scaleX
			y2 := (boxCy + boxH/2) * detSz * scaleY
			lmBase := i * 10
			var lm [5][2]float32
			for p := 0; p < 5; p++ {
				// Decode landmark coordinates from anchor.
				// MobileNet bakes variance[0] into the output, so we
				// decode with raw * prior_size only.
				// ResNet50 does NOT bake variance, so we multiply by
				// variance[0] * prior_size.
				var lx, ly float32
				if landmarkVarianceBaked {
					lx = pr.cx + lmData[lmBase+2*p+0]*pr.sx
					ly = pr.cy + lmData[lmBase+2*p+1]*pr.sy
				} else {
					lx = pr.cx + lmData[lmBase+2*p+0]*retinaVariance[0]*pr.sx
					ly = pr.cy + lmData[lmBase+2*p+1]*retinaVariance[0]*pr.sy
				}
				lm[p][0] = lx * detSz * scaleX
				lm[p][1] = ly * detSz * scaleY
			}
			if !loggedTop {
				loggedTop = true
				slog.Default().Debug("top-scoring anchor",
					"stride", g.stride,
					"score", prob,
					"anchor_cx", pr.cx, "anchor_cy", pr.cy,
					"anchor_sx", pr.sx, "anchor_sy", pr.sy,
					"raw_lx0", lmData[lmBase+0], "raw_ly0", lmData[lmBase+1],
					"raw_lx1", lmData[lmBase+2], "raw_ly1", lmData[lmBase+3],
					"raw_lx2", lmData[lmBase+4], "raw_ly2", lmData[lmBase+5],
					"raw_lx3", lmData[lmBase+6], "raw_ly3", lmData[lmBase+7],
					"raw_lx4", lmData[lmBase+8], "raw_ly4", lmData[lmBase+9],
					"lm0", fmt.Sprintf("(%g,%g)", lm[0][0], lm[0][1]),
					"lm1", fmt.Sprintf("(%g,%g)", lm[1][0], lm[1][1]),
					"lm2", fmt.Sprintf("(%g,%g)", lm[2][0], lm[2][1]),
					"lm3", fmt.Sprintf("(%g,%g)", lm[3][0], lm[3][1]),
					"lm4", fmt.Sprintf("(%g,%g)", lm[4][0], lm[4][1]),
				)
			}
			bbW := x2 - x1
			bbH := y2 - y1
			if bbW < minFacePx || bbH < minFacePx {
				continue
			}
			// Quality filter: skip faces with extreme aspect ratios
			// (likely false positives from body parts or objects)
			aspect := bbW / bbH
			if aspect < 0.5 || aspect > 2.0 {
				continue
			}
			// Quality filter: skip faces where landmarks are outside bbox
			// (indicates misaligned detection)
			if !landmarksInBBox(lm, x1, y1, x2, y2) {
				continue
			}
			allFaces = append(allFaces, face{
				X1: x1, Y1: y1, X2: x2, Y2: y2,
				Landmarks: lm, Score: prob,
			})
		}
		slog.Default().Debug("decodeRetinaFaces stride stats",
			"stride", g.stride, "max", strideMax, "above_thr", strideAbove, "threshold", confThr)
	}
	return nmsFaces(allFaces, iouThr)
}

// logTopNCandidates logs the top-N scoring anchors BEFORE threshold
// and NMS, with their center coordinates in the ORIGINAL image.
// This is the single most useful diagnostic: if the top scores are
// NOT near the real face, the model weights or preprocessing are wrong.
func logTopNCandidates(groups []retinaOut, orig image.Rectangle, n int) {
	type scored struct {
		stride int
		score  float32
		cx, cy float32 // original image pixel coords
	}
	var all []scored

	for _, g := range groups {
		if g.scores == nil || g.outputSize == 0 {
			continue
		}
		detSz := float32(detSizeFromGroup(g))
		priors := priorsForStride(g.stride, int(detSz))
		scaleX := float32(orig.Dx()) / detSz
		scaleY := float32(orig.Dy()) / detSz
		scores := g.scores.GetData()
		for i, pr := range priors {
			if i >= len(scores) {
				break
			}
			all = append(all, scored{
				stride: g.stride,
				score:  scores[i],
				cx:     pr.cx * detSz * scaleX,
				cy:     pr.cy * detSz * scaleY,
			})
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	if len(all) > n {
		all = all[:n]
	}
	for i, a := range all {
		slog.Default().Debug("top candidate",
			"rank", i, "stride", a.stride,
			"score", a.score,
			"orig_cx", a.cx, "orig_cy", a.cy)
	}
}

func (g retinaOut) landmarkssafe() []float32 {
	if g.landmarks == nil {
		return nil
	}
	return g.landmarks.GetData()
}

// detSizeFromGroup returns the detector's input size — stored as
// (stride^2) * (outputSize / (inputSize/stride))^2 back to inputSize.
// Practically: inputSize = stride * h. We don't have it directly
// here, so the caller must pass pre-computed scale factors via the
// face struct (already scaled). This helper is a no-op kept for
// future use.
func detSizeFromGroup(g retinaOut) float32 {
	if g.outputSize == 0 || g.stride == 0 {
		return 0
	}
	// outputSize = hw*hw (grid cells only, no AnchorsPerCell multiplier);
	// sqrt gives the grid dimension h, and h*stride == detInputSize.
	gs := int(math.Sqrt(float64(g.outputSize)))
	return float32(gs * g.stride)
}

// landmarksInBBox checks if all 5 landmarks are within the bounding box
// (with some margin for edge cases). Returns false if landmarks suggest
// a misaligned detection.
func landmarksInBBox(lm [5][2]float32, x1, y1, x2, y2 float32) bool {
	margin := float32(0.1) // 10% margin
	w := x2 - x1
	h := y2 - y1
	for _, p := range lm {
		if p[0] < x1-w*margin || p[0] > x2+w*margin {
			return false
		}
		if p[1] < y1-h*margin || p[1] > y2+h*margin {
			return false
		}
	}
	return true
}

// writeAlignedBatchToTensor packs multiple 112x112 aligned faces into a
// batched ArcFace input tensor [batchSize, 3, 112, 112] in BGR order.
// Returns the number of faces actually written (may be < len(aligned)
// if the tensor is full).
func writeAlignedBatchToTensor(aligned []*image.RGBA, dst *ort.Tensor[float32]) int {
	if dst == nil || len(aligned) == 0 {
		return 0
	}
	tData := dst.GetData()
	batchSz := len(tData) / (3 * ArcFaceInputSize * ArcFaceInputSize)
	if batchSz > len(aligned) {
		batchSz = len(aligned)
	}
	channels := 3 * ArcFaceInputSize * ArcFaceInputSize
	for i := 0; i < batchSz; i++ {
		rgba := aligned[i]
		pix := rgba.Pix
		base := i * channels
		for y := 0; y < ArcFaceInputSize; y++ {
			for x := 0; x < ArcFaceInputSize; x++ {
				off := (y*rgba.Stride + x*4)
				r8 := float32(pix[off])
				g8 := float32(pix[off+1])
				b8 := float32(pix[off+2])
				tData[base+y*ArcFaceInputSize+x] = (b8 / 127.5) - 1.0
				tData[base+ArcFaceInputSize*ArcFaceInputSize+y*ArcFaceInputSize+x] = (g8 / 127.5) - 1.0
				tData[base+2*ArcFaceInputSize*ArcFaceInputSize+y*ArcFaceInputSize+x] = (r8 / 127.5) - 1.0
			}
		}
	}
	return batchSz
}

// nmsFaces sorts by score and removes IoU-overlapping lower-scored
// boxes; returns the survivors. O(n²) where n = pre-NMS detections.
// In practice n is small (10–50) after quality filters (minFacePx,
// aspect ratio, landmarksInBBox) prune the 16800 raw anchors. If
// future models produce more surviving detections, consider switching
// to a sorted-by-score greedy NMS with early termination.
func nmsFaces(in []face, iouThr float32) []face {
	if len(in) == 0 {
		return in
	}
	sort.SliceStable(in, func(i, j int) bool { return in[i].Score > in[j].Score })
	kept := make([]face, 0, len(in))
	for _, f := range in {
		drop := false
		for _, k := range kept {
			if iou(f, k) > iouThr {
				drop = true
				break
			}
		}
		if !drop {
			kept = append(kept, f)
		}
	}
	return kept
}

func iou(a, b face) float32 {
	ix1 := math.Max(float64(a.X1), float64(b.X1))
	iy1 := math.Max(float64(a.Y1), float64(b.Y1))
	ix2 := math.Min(float64(a.X2), float64(b.X2))
	iy2 := math.Min(float64(a.Y2), float64(b.Y2))
	if ix2 <= ix1 || iy2 <= iy1 {
		return 0
	}
	inter := (ix2 - ix1) * (iy2 - iy1)
	areaA := (float64(a.X2) - float64(a.X1)) * (float64(a.Y2) - float64(a.Y1))
	areaB := (float64(b.X2) - float64(b.X1)) * (float64(b.Y2) - float64(b.Y1))
	return float32(inter / (areaA + areaB - inter))
} // --- 5-point affine alignment to 112x112 ---

// alignFaceFromBox takes a decoded face and produces a 112×112 RGB
// image that ArcFace can ingest.
//
// One path only: 5-point affine alignment to arcFaceRef. The
// previous version tried a 1.5×-bbox square crop as a fallback when
// landmarks clustered near the centre or the affine solver went
// singular. That fallback produced 112×112 patches that ArcFace
// embedded to near-identical vectors regardless of whose face was
// in the photo — at default threshold 0.55 the false-positive rate
// was 100× (2000+ matches in a 25k-photo archive, most of them
// random textures the detector NMS'd in on weak signal). Real
// faces always have well-spread landmarks, so degenerate cases
// are the detector confusing a non-face patch for a face; the
// correct response is to skip, NOT to embed garbage.
//
// extractor.extractWithDebug already treats any error returned
// here as "skip this face, continue with the rest", so we don't
// need a sentinel return value.
func alignFaceFromBox(src image.Image, f face) (*image.RGBA, error) {
	if src == nil || src.Bounds().Empty() {
		return nil, errors.New("alignFaceFromBox: empty source")
	}
	aligned, err := alignFaceTo112(src, f.Landmarks)
	if err != nil {
		slog.Default().Debug("align failed (no bbox fallback), skipping face",
			"err", err, "score", f.Score)
		return nil, err
	}
	return aligned, nil
}

// alignFaceTo112 solves the 2x3 affine matrix mapping srcPts (in
// original image coordinates) to arcFaceRef (canonical 112x112) and
// bilinearly samples into a fresh 112x112 RGB image.
// Uses type-switch for direct Pix access when possible.
func alignFaceTo112(src image.Image, srcPts [5][2]float32) (*image.RGBA, error) {
	if src.Bounds().Empty() {
		return nil, errors.New("align: empty source image")
	}
	a, b, c, d, e, f, err := solveAffine(srcPts, arcFaceRef)
	if err != nil {
		return nil, fmt.Errorf("align: solve affine: %w", err)
	}
	out := image.NewRGBA(image.Rect(0, 0, ArcFaceInputSize, ArcFaceInputSize))
	det := a*e - b*d
	if det == 0 {
		return nil, errors.New("align: degenerate affine")
	}
	invDet := 1.0 / det
	srcBounds := src.Bounds()

	// Type-switch for direct Pix access — avoids interface dispatch per pixel
	switch v := src.(type) {
	case *image.RGBA:
		alignFaceRGBADirect(v, out, srcBounds, a, b, c, d, e, f, invDet)
	case *image.NRGBA:
		alignFaceNRGBADirect(v, out, srcBounds, a, b, c, d, e, f, invDet)
	case *image.YCbCr:
		alignFaceYCbCrDirect(v, out, srcBounds, a, b, c, d, e, f, invDet)
	default:
		alignFaceGenericDirect(src, out, srcBounds, a, b, c, d, e, f, invDet)
	}
	return out, nil
}

// alignFaceRGBADirect aligns with direct Pix access for *image.RGBA source.
func alignFaceRGBADirect(src *image.RGBA, out *image.RGBA, srcBounds image.Rectangle, a, b, c, d, e, f, invDet float64) {
	srcPix := src.Pix
	srcStride := src.Stride
	outPix := out.Pix
	outStride := out.Stride
	minX := float64(srcBounds.Min.X)
	minY := float64(srcBounds.Min.Y)
	maxX := float64(srcBounds.Max.X - 2)
	maxY := float64(srcBounds.Max.Y - 2)
	for y := 0; y < ArcFaceInputSize; y++ {
		for x := 0; x < ArcFaceInputSize; x++ {
			sx := (e*(float64(x)-c) - b*(float64(y)-f)) * invDet
			sy := (-d*(float64(x)-c) + a*(float64(y)-f)) * invDet
			x0 := clampInt(int(math.Floor(sx)), int(minX), int(maxX))
			y0 := clampInt(int(math.Floor(sy)), int(minY), int(maxY))
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			off00 := y0*srcStride + x0*4
			off01 := off00 + 4
			off10 := off00 + srcStride
			off11 := off10 + 4
			r := lerp(lerp(float64(srcPix[off00])*257, float64(srcPix[off01])*257, fx),
				lerp(float64(srcPix[off10])*257, float64(srcPix[off11])*257, fx), fy)
			g := lerp(lerp(float64(srcPix[off00+1])*257, float64(srcPix[off01+1])*257, fx),
				lerp(float64(srcPix[off10+1])*257, float64(srcPix[off11+1])*257, fx), fy)
			bv := lerp(lerp(float64(srcPix[off00+2])*257, float64(srcPix[off01+2])*257, fx),
				lerp(float64(srcPix[off10+2])*257, float64(srcPix[off11+2])*257, fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(bv) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// alignFaceNRGBADirect aligns with direct Pix access for *image.NRGBA source.
func alignFaceNRGBADirect(src *image.NRGBA, out *image.RGBA, srcBounds image.Rectangle, a, b, c, d, e, f, invDet float64) {
	srcPix := src.Pix
	srcStride := src.Stride
	outPix := out.Pix
	outStride := out.Stride
	minX := float64(srcBounds.Min.X)
	minY := float64(srcBounds.Min.Y)
	maxX := float64(srcBounds.Max.X - 2)
	maxY := float64(srcBounds.Max.Y - 2)
	for y := 0; y < ArcFaceInputSize; y++ {
		for x := 0; x < ArcFaceInputSize; x++ {
			sx := (e*(float64(x)-c) - b*(float64(y)-f)) * invDet
			sy := (-d*(float64(x)-c) + a*(float64(y)-f)) * invDet
			x0 := clampInt(int(math.Floor(sx)), int(minX), int(maxX))
			y0 := clampInt(int(math.Floor(sy)), int(minY), int(maxY))
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			off00 := y0*srcStride + x0*4
			off01 := off00 + 4
			off10 := off00 + srcStride
			off11 := off10 + 4
			r := lerp(lerp(float64(srcPix[off00])*257, float64(srcPix[off01])*257, fx),
				lerp(float64(srcPix[off10])*257, float64(srcPix[off11])*257, fx), fy)
			g := lerp(lerp(float64(srcPix[off00+1])*257, float64(srcPix[off01+1])*257, fx),
				lerp(float64(srcPix[off10+1])*257, float64(srcPix[off11+1])*257, fx), fy)
			bv := lerp(lerp(float64(srcPix[off00+2])*257, float64(srcPix[off01+2])*257, fx),
				lerp(float64(srcPix[off10+2])*257, float64(srcPix[off11+2])*257, fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(bv) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// alignFaceYCbCrDirect aligns with At() for *image.YCbCr source.
func alignFaceYCbCrDirect(src *image.YCbCr, out *image.RGBA, srcBounds image.Rectangle, a, b, c, d, e, f, invDet float64) {
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < ArcFaceInputSize; y++ {
		for x := 0; x < ArcFaceInputSize; x++ {
			sx := (e*(float64(x)-c) - b*(float64(y)-f)) * invDet
			sy := (-d*(float64(x)-c) + a*(float64(y)-f)) * invDet
			x0 := clampInt(int(math.Floor(sx)), srcBounds.Min.X, srcBounds.Max.X-2)
			y0 := clampInt(int(math.Floor(sy)), srcBounds.Min.Y, srcBounds.Max.Y-2)
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			r00, g00, b00, _ := src.At(x0, y0).RGBA()
			r01, g01, b01, _ := src.At(x0+1, y0).RGBA()
			r10, g10, b10, _ := src.At(x0, y0+1).RGBA()
			r11, g11, b11, _ := src.At(x0+1, y0+1).RGBA()
			r := lerp(lerp(float64(r00), float64(r01), fx), lerp(float64(r10), float64(r11), fx), fy)
			g := lerp(lerp(float64(g00), float64(g01), fx), lerp(float64(g10), float64(g11), fx), fy)
			bv := lerp(lerp(float64(b00), float64(b01), fx), lerp(float64(b10), float64(b11), fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(bv) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// alignFaceGenericDirect is the fallback for unknown image types.
func alignFaceGenericDirect(src image.Image, out *image.RGBA, srcBounds image.Rectangle, a, b, c, d, e, f, invDet float64) {
	outPix := out.Pix
	outStride := out.Stride
	for y := 0; y < ArcFaceInputSize; y++ {
		for x := 0; x < ArcFaceInputSize; x++ {
			sx := (e*(float64(x)-c) - b*(float64(y)-f)) * invDet
			sy := (-d*(float64(x)-c) + a*(float64(y)-f)) * invDet
			x0 := clampInt(int(math.Floor(sx)), srcBounds.Min.X, srcBounds.Max.X-2)
			y0 := clampInt(int(math.Floor(sy)), srcBounds.Min.Y, srcBounds.Max.Y-2)
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			r00, g00, b00, _ := src.At(x0, y0).RGBA()
			r01, g01, b01, _ := src.At(x0+1, y0).RGBA()
			r10, g10, b10, _ := src.At(x0, y0+1).RGBA()
			r11, g11, b11, _ := src.At(x0+1, y0+1).RGBA()
			r := lerp(lerp(float64(r00), float64(r01), fx), lerp(float64(r10), float64(r11), fx), fy)
			g := lerp(lerp(float64(g00), float64(g01), fx), lerp(float64(g10), float64(g11), fx), fy)
			bv := lerp(lerp(float64(b00), float64(b01), fx), lerp(float64(b10), float64(b11), fx), fy)
			outOff := y*outStride + x*4
			outPix[outOff] = uint8(clampUint16(r) / 256)
			outPix[outOff+1] = uint8(clampUint16(g) / 256)
			outPix[outOff+2] = uint8(clampUint16(bv) / 256)
			outPix[outOff+3] = 0xff
		}
	}
}

// solveAffine: 2x3 affine M = [a b c; d e f] s.t. refs[i] = M * srcPts[i].
// 10 equations (5 pts × 2), 6 unknowns — solved via least squares.
func solveAffine(srcPts [5][2]float32, refs [5][2]float32) (a, b, c, d, e, f float64, err error) {
	A := make([]float64, 60) // 10 rows × 6 cols, row-major
	Bx := make([]float64, 10)
	By := make([]float64, 10)
	for i := 0; i < 5; i++ {
		sx := float64(srcPts[i][0])
		sy := float64(srcPts[i][1])
		rx := float64(refs[i][0])
		ry := float64(refs[i][1])
		A[(2*i)*6+0] = sx
		A[(2*i)*6+1] = sy
		A[(2*i)*6+2] = 1
		Bx[2*i] = rx
		A[(2*i+1)*6+3] = sx
		A[(2*i+1)*6+4] = sy
		A[(2*i+1)*6+5] = 1
		By[2*i+1] = ry
	}
	AtA := make([]float64, 36)
	AtBx := make([]float64, 6)
	AtBy := make([]float64, 6)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			s := 0.0
			for k := 0; k < 10; k++ {
				s += A[k*6+i] * A[k*6+j]
			}
			AtA[i*6+j] = s
		}
		s := 0.0
		t := 0.0
		for k := 0; k < 10; k++ {
			s += A[k*6+i] * Bx[k]
			t += A[k*6+i] * By[k]
		}
		AtBx[i] = s
		AtBy[i] = t
	}
	xa, err := solveLinear6x6(AtA, AtBx)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf("x: %w", err)
	}
	xb, err := solveLinear6x6(AtA, AtBy)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, fmt.Errorf("y: %w", err)
	}
	return xa[0], xa[1], xa[2], xb[3], xb[4], xb[5], nil
}

func solveLinear6x6(M []float64, b []float64) ([]float64, error) {
	const n = 6
	A := make([]float64, n*n)
	copy(A, M)
	y := make([]float64, n)
	copy(y, b)
	for i := 0; i < n; i++ {
		piv := i
		maxVal := math.Abs(A[i*n+i])
		for k := i + 1; k < n; k++ {
			if v := math.Abs(A[k*n+i]); v > maxVal {
				maxVal = v
				piv = k
			}
		}
		if maxVal < 1e-9 {
			return nil, errors.New("singular matrix")
		}
		if piv != i {
			for k := 0; k < n; k++ {
				A[i*n+k], A[piv*n+k] = A[piv*n+k], A[i*n+k]
			}
			y[i], y[piv] = y[piv], y[i]
		}
		p := A[i*n+i]
		for k := i; k < n; k++ {
			A[i*n+k] /= p
		}
		y[i] /= p
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			f := A[j*n+i]
			for k := i; k < n; k++ {
				A[j*n+k] -= f * A[i*n+k]
			}
			y[j] -= f * y[i]
		}
	}
	return y, nil
}

// writeAlignedToTensor packs a 112x112 RGB uint8 source into ArcFace
// input as CHW float32 BGR normalized (x/127.5) - 1. ArcFace models
// from insightface expect BGR channel order (OpenCV convention).
func writeAlignedToTensor(src image.Image, dst *ort.Tensor[float32]) error {
	if dst == nil {
		return errors.New("write aligned: nil tensor")
	}
	b := src.Bounds()
	if b.Dx() != ArcFaceInputSize || b.Dy() != ArcFaceInputSize {
		return errors.New("write aligned: unexpected source size")
	}
	tData := dst.GetData()
	if len(tData) != 3*ArcFaceInputSize*ArcFaceInputSize {
		return errors.New("write aligned: tensor buffer size mismatch")
	}
	// Direct Pix access — alignFaceTo112 always returns *image.RGBA
	rgba := src.(*image.RGBA)
	pix := rgba.Pix
	stride := ArcFaceInputSize * ArcFaceInputSize
	for y := 0; y < ArcFaceInputSize; y++ {
		for x := 0; x < ArcFaceInputSize; x++ {
			off := (y*rgba.Stride + x*4)
			r8 := float32(pix[off])   // R
			g8 := float32(pix[off+1]) // G
			b8 := float32(pix[off+2]) // B
			tData[y*ArcFaceInputSize+x] = (b8 / 127.5) - 1.0
			tData[stride+y*ArcFaceInputSize+x] = (g8 / 127.5) - 1.0
			tData[2*stride+y*ArcFaceInputSize+x] = (r8 / 127.5) - 1.0
		}
	}
	return nil
}

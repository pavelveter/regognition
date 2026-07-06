// Package embedder — ONNX implementation.
//
// This file wires retina-face-style face detection and arc-face-style
// recognition into the single Embedder interface the pipeline sees.
// Implementation helpers (preprocessing, decode, alignment) live in
// onnx_runtime.go.
//
// THE TWO BIG ASSUMPTIONS (configurable in ONNXOptions):
//
//   1. The detector has 9 output tensors, three per stride (8/16/32),
//      each representing scores (2*A channels), bboxes (4*A), and
//      landmarks (10*A). A — anchors per cell — is hard-coded at 2 for
//      retinaface_mnet025v2.
//   2. The recognizer has exactly one output tensor of shape [1, 512].
//
// If your model uses different layouts, set ONNXOptions.DetectorInput/
// OutputNames and EmbedderInput/OutputName explicitly. Open the ONNX in
// Netron to verify names and shapes.
//
// Output embeddings are L2-normalized (cosine contract: same as stub).

package embedder

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

func init() {
	// Force single-threaded matmul for deterministic inference output.
	// ORT's multi-threaded OMP/MKL floating-point ordering is
	// non-deterministic, causing score flicker between runs on the
	// same input. With NumThreads=1, scores are reproducible and the
	// same threshold consistently produces the same NMS result.
	// Set BEFORE ort.InitializeEnvironment() so the C++ runtime picks
	// it up at construction.
	os.Setenv("OMP_NUM_THREADS", "1")
	os.Setenv("MKL_NUM_THREADS", "1")
}

// Constants for the canonical retinaface_mnet025v2 + standard ArcFace
// recognizer. Override via ONNXOptions when your model differs.
const (
	AnchorsPerCell = 2 // retinaface_mnet025v2 always exports 2 anchors per cell

	ArcFaceInputSize = 112
	ArcFaceDim       = 512
)

// arcFaceRef is the canonical 5-point ArcFace reference, mapped to a
// 112x112 template. Used by onnx_runtime.go's alignFaceTo112.
var arcFaceRef = [5][2]float32{
	{30.2946, 51.6963},
	{65.5318, 51.5014},
	{48.0252, 71.7366},
	{33.5493, 92.3655},
	{62.7299, 92.2041},
}

// ONNXOptions configures NewONNX. Defaults are filled in for fields
// left at zero, except for paths which are required.
//
// Detector output names MUST match your model's I/O. The default set
// is the standard InsightFace/buffalo_sc convention; if your ONNX
// uses different names (e.g. "scores"/"boxes"/"landmarks"), override.
type ONNXOptions struct {
	LibPath      string  // absolute or relative path to libonnxruntime.{dylib,so,dll}
	DetInputSize int     // default 640
	Workers      int     // default 4
	DetThreshold float32 // face confidence threshold, default 0.5
	NMSIoU       float32 // NMS IoU threshold, default 0.4

	// Detector I/O. If left empty, defaults are populated by NewONNX.
	DetectorInputName string // PyTorch ONNX exporter names the canvas input "input.1"
	DetectorBaseNames []string
	DetectorSuffixes  []string
	// DetectorOutputNames is the FLAT list of 9 output tensor names in
	// stride-major × type-minor order: scores8, boxes8, landm8,
	// scores16, boxes16, landm16, scores32, boxes32, landm32.
	//
	// PyTorch's ONNX exporter assigns numeric names in declaration
	// order. The canonical biubug6/Pytorch_Retinaface export declares
	// them in exactly that order — see the default in
	// fillONNXOptionDefaults. If your detector uses semantic names
	// (e.g. "face_r8_cls") instead, override via ONNXOptions.
	DetectorOutputNames []string

	// DetectorFormat controls how detector outputs are structured:
	//   - "split" (default): 9 output tensors, 3 per stride (scores, boxes, landmarks)
	//   - "concat": 3 output tensors concatenated across strides (ResNet50 style)
	DetectorFormat string

	// LandmarkVarianceBaked indicates whether the ONNX export bakes
	// variance[0] into landmark outputs. MobileNet does (true), ResNet50
	// does not (false). When false, variance[0] is applied during decode.
	LandmarkVarianceBaked bool

	// Recognizer I/O.
	EmbedderInputName  string // default "input"
	EmbedderOutputName string // default "fc1"

	// Execution Providers
	UseCoreML bool // enable CoreML EP (macOS only)

	// Batching: max faces per ArcFace inference call.
	// Higher = fewer inference calls, more memory. Default 1 (no batching).
	ArcBatchSize int
}

const (
	DefaultDetInputSize = 640
	DefaultWorkers      = 4
	// 0.5 — retinaface_mnet025_v2 / ResNet50 baseline. Override via
	// INI / --detector-threshold.
	DefaultDetThreshold = 0.5
	DefaultNMSIoU       = 0.4
	DefaultBaseNames    = "face_r8"
)

// retinaOut groups the three detector output tensors for one stride,
// paired with the stride itself so the decoder doesn't need
// introspection (yalue's *Tensor has no exported Shape).
type retinaOut struct {
	stride     int
	scores     *ort.Tensor[float32]
	bboxes     *ort.Tensor[float32]
	landmarks  *ort.Tensor[float32]
	outputSize int // HW = stride cells-per-side squared (H == W)
}

// sessionPair is one worker's pre-allocated detector + recognizer
// resources. Workers pull a pair from the pool, hold it for the
// duration of one Extract call, then return it.
type sessionPair struct {
	detSession *ort.AdvancedSession
	arcSession *ort.AdvancedSession
	detInput   *ort.Tensor[float32]
	arcInput   *ort.Tensor[float32] // shape: [batchSize, 3, 112, 112]
	detOuts    []retinaOut          // 3 entries (per stride)
	arcOutput  *ort.Tensor[float32] // shape: [batchSize, 512]
	arcBatchSz int                  // actual batch size used

	// For concat-format detectors (ResNet50): raw output tensors
	// that need post-inference splitting into detOuts.
	concatScores *ort.Tensor[float32] // [1, 16800, 2]
	concatBboxes *ort.Tensor[float32] // [1, 16800, 4]
	concatLm     *ort.Tensor[float32] // [1, 16800, 10]

	// Pre-allocated split tensors for concat-format detectors.
	// Reused across calls to avoid per-call allocations.
	splitScores [3]*ort.Tensor[float32] // [ahw, 1] per stride
	splitBboxes [3]*ort.Tensor[float32] // [ahw, 4] per stride
	splitLm     [3]*ort.Tensor[float32] // [ahw, 10] per stride
}

// ONNXEmbedder is the ONNX-backed Embedder. Built around a buffered
// channel of per-worker session pairs so concurrent Extract calls do
// NOT contend on a shared ONNX Session.
type ONNXEmbedder struct {
	pool       chan *sessionPair
	opts       ONNXOptions
	detInputSz int

	initOnce  sync.Once
	initErr   error
	closeOnce sync.Once
	closeMu   sync.Mutex
	closed    bool
}

// NewONNX loads both ONNX files, builds per-worker session pairs,
// and returns an Embedder ready for concurrent use.
//
// If anything fails after partial success, all allocated resources
// are released before returning.
func NewONNX(detPath, arcPath string, opts ONNXOptions) (*ONNXEmbedder, error) {
	if detPath == "" || arcPath == "" {
		return nil, errors.New("onnx: detector and embedder model paths required")
	}
	for _, p := range []string{detPath, arcPath} {
		if abs, err := filepath.Abs(p); err == nil {
			_ = abs // keep abs for clearer error messages if loading fails
		}
	}

	fillONNXOptionDefaults(&opts)

	e := &ONNXEmbedder{
		pool:       make(chan *sessionPair, opts.Workers),
		opts:       opts,
		detInputSz: opts.DetInputSize,
	}

	// Set the path FIRST, before any ort.* call.
	if opts.LibPath != "" {
		ort.SetSharedLibraryPath(opts.LibPath)
	}

	var initOK bool
	e.initOnce.Do(func() {
		initOK = true
		if err := ort.InitializeEnvironment(); err != nil {
			initOK = false
			e.initErr = fmt.Errorf("onnx: init env: %w", err)
		}
	})
	if !initOK {
		return nil, e.initErr
	}

	for i := 0; i < opts.Workers; i++ {
		sp, err := buildSessionPair(detPath, arcPath, opts)
		if err != nil {
			e.closePoolAndDestroy()
			return nil, fmt.Errorf("onnx: build session pair %d: %w", i, err)
		}
		e.pool <- sp
	}
	return e, nil
}

func fillONNXOptionDefaults(o *ONNXOptions) {
	if o.DetInputSize == 0 {
		o.DetInputSize = DefaultDetInputSize
	}
	if o.Workers == 0 {
		o.Workers = DefaultWorkers
	}
	if o.DetThreshold == 0 {
		o.DetThreshold = DefaultDetThreshold
	}
	if o.NMSIoU == 0 {
		o.NMSIoU = DefaultNMSIoU
	}
	if o.DetectorInputName == "" {
		// PyTorch's ONNX exporter labels the canvas input as "input.1"
		// (the first named input slot). Verify with onnx.load on
		// your .onnx model.
		o.DetectorInputName = "input.1"
	}
	if o.EmbedderInputName == "" {
		// arcface_r50_fp16 export: PyTorch's ONNX exporter names
		// the canvas input "input.1" (the first slot usually holds
		// the placeholder). Verify with onnx.load on your .onnx.
		o.EmbedderInputName = "input.1"
	}
	if o.EmbedderOutputName == "" {
		// arcface_r50_fp16 export emits a single 512-d vector
		// named "516" in declaration order. Override via INI /
		// --embedder-output.
		o.EmbedderOutputName = "516"
	}

	// Detector format and output names depend on the model type.
	if o.DetectorFormat == "" {
		// Auto-detect: if 3 output names → concat, if 9 → split
		if len(o.DetectorOutputNames) == 3 {
			o.DetectorFormat = "concat"
		} else {
			o.DetectorFormat = "split"
		}
	}

	if o.DetectorFormat == "concat" {
		// ResNet50-style: 3 concatenated outputs
		if len(o.DetectorOutputNames) == 0 {
			o.DetectorOutputNames = []string{"bbox", "confidence", "landmark"}
		}
		if !o.LandmarkVarianceBaked {
			// ResNet50 does NOT bake variance into landmarks
			o.LandmarkVarianceBaked = false
		}
	} else {
		// MobileNet-style: 9 split outputs
		if len(o.DetectorBaseNames) == 0 {
			o.DetectorBaseNames = []string{"face_r8", "face_r16", "face_r32"}
		}
		if len(o.DetectorSuffixes) == 0 {
			o.DetectorSuffixes = []string{"cls", "bbox", "landm"}
		}
		if len(o.DetectorOutputNames) == 0 {
			// Default for the biubug6/Pytorch_Retinaface MobileNetV2 export
			// (retinaface_mnet025_v2): PyTorch's ONNX exporter assigns
			// numeric output names in graph declaration order. The export
			// declares nine outputs in stride-major × type-minor order, so
			// strides 8/16/32 × (scores, boxes, landm) lay out as:
			// 443, 446, 449, 468, 471, 474, 493, 496, 499.
			o.DetectorOutputNames = []string{
				"443", "446", "449", // stride 8: scores, boxes, landm
				"468", "471", "474", // stride 16
				"493", "496", "499", // stride 32
			}
		}
	}
}

func (e *ONNXEmbedder) closePoolAndDestroy() {
	e.closeMu.Lock()
	if e.closed {
		e.closeMu.Unlock()
		return
	}
	e.closed = true
	e.closeMu.Unlock()
	if e.pool == nil {
		return
	}
	close(e.pool)
	for sp := range e.pool {
		destroySessionPair(sp)
	}
}

// buildSessionPair is the heavy constructor — creates detector output
// tensors, 1 recognizer input + output, and 2 sessions. Supports both
// split (9 tensors) and concat (3 tensors) detector output formats.
func buildSessionPair(detPath, arcPath string, opts ONNXOptions) (*sessionPair, error) {
	// --- Session options (CoreML EP if requested) ---
	var sessionOpts *ort.SessionOptions
	if opts.UseCoreML {
		so, err := ort.NewSessionOptions()
		if err != nil {
			return nil, fmt.Errorf("session options: %w", err)
		}
		coreMLOpts := map[string]string{
			"MLComputeUnits": "ALL",
		}
		if err := so.AppendExecutionProviderCoreMLV2(coreMLOpts); err != nil {
			so.Destroy()
			return nil, fmt.Errorf("coreml ep: %w", err)
		}
		sessionOpts = so
	}

	// --- Detector input ---
	detInShape := ort.NewShape(int64(1), int64(3), int64(opts.DetInputSize), int64(opts.DetInputSize))
	detIn, err := ort.NewTensor(detInShape, make([]float32, 3*opts.DetInputSize*opts.DetInputSize))
	if err != nil {
		if sessionOpts != nil {
			sessionOpts.Destroy()
		}
		return nil, fmt.Errorf("det input tensor: %w", err)
	}

	var detSession *ort.AdvancedSession
	var detGroups []retinaOut
	var detOutputs []ort.Value
	var concatScores, concatBboxes, concatLm *ort.Tensor[float32]
	var splitScores [3]*ort.Tensor[float32]
	var splitBboxes [3]*ort.Tensor[float32]
	var splitLm [3]*ort.Tensor[float32]

	if opts.DetectorFormat == "concat" {
		// ResNet50-style: 3 concatenated outputs [1, 16800, C]
		detSession, concatScores, concatBboxes, concatLm, err = buildConcatSession(
			detPath, opts, detIn, sessionOpts,
		)
		if err != nil {
			detIn.Destroy()
			if sessionOpts != nil {
				sessionOpts.Destroy()
			}
			return nil, err
		}
		// Create per-stride output groups that will be filled after inference
		detGroups = makeSplitGroups(opts.DetInputSize)
		// Pre-allocate split tensors for reuse across calls
		strides := []int{8, 16, 32}
		for i, s := range strides {
			hw := opts.DetInputSize / s
			ahw := AnchorsPerCell * hw * hw
			splitScores[i], _ = ort.NewEmptyTensor[float32](ort.NewShape(int64(ahw), 1))
			splitBboxes[i], _ = ort.NewEmptyTensor[float32](ort.NewShape(int64(ahw), 4))
			splitLm[i], _ = ort.NewEmptyTensor[float32](ort.NewShape(int64(ahw), 10))
		}
	} else {
		// MobileNet-style: 9 split outputs [A*HW, C]
		if len(opts.DetectorOutputNames) != 9 {
			detIn.Destroy()
			if sessionOpts != nil {
				sessionOpts.Destroy()
			}
			return nil, fmt.Errorf("split format requires 9 DetectorOutputNames, got %d", len(opts.DetectorOutputNames))
		}
		detSession, detOutputs, detGroups, err = buildSplitSession(
			detPath, opts, detIn, sessionOpts,
		)
		if err != nil {
			detIn.Destroy()
			if sessionOpts != nil {
				sessionOpts.Destroy()
			}
			return nil, err
		}
	}

	// --- Recognizer (with optional batching) ---
	batchSz := opts.ArcBatchSize
	if batchSz < 1 {
		batchSz = 1
	}
	arcInShape := ort.NewShape(int64(batchSz), int64(3), int64(ArcFaceInputSize), int64(ArcFaceInputSize))
	arcIn, err := ort.NewTensor(arcInShape, make([]float32, batchSz*3*ArcFaceInputSize*ArcFaceInputSize))
	if err != nil {
		destroyDetector(detSession, detIn, detOutputs)
		if concatScores != nil {
			concatScores.Destroy()
		}
		if concatBboxes != nil {
			concatBboxes.Destroy()
		}
		if concatLm != nil {
			concatLm.Destroy()
		}
		if sessionOpts != nil {
			sessionOpts.Destroy()
		}
		return nil, fmt.Errorf("arc input tensor: %w", err)
	}
	arcOutShape := ort.NewShape(int64(batchSz), int64(ArcFaceDim))
	arcOut, err := ort.NewEmptyTensor[float32](arcOutShape)
	if err != nil {
		destroyDetector(detSession, detIn, detOutputs)
		if concatScores != nil {
			concatScores.Destroy()
		}
		if concatBboxes != nil {
			concatBboxes.Destroy()
		}
		if concatLm != nil {
			concatLm.Destroy()
		}
		arcIn.Destroy()
		if sessionOpts != nil {
			sessionOpts.Destroy()
		}
		return nil, fmt.Errorf("arc output tensor: %w", err)
	}
	arcSession, err := ort.NewAdvancedSession(
		arcPath,
		[]string{opts.EmbedderInputName},
		[]string{opts.EmbedderOutputName},
		[]ort.Value{arcIn},
		[]ort.Value{arcOut},
		sessionOpts,
	)
	if err != nil {
		destroyDetector(detSession, detIn, detOutputs)
		if concatScores != nil {
			concatScores.Destroy()
		}
		if concatBboxes != nil {
			concatBboxes.Destroy()
		}
		if concatLm != nil {
			concatLm.Destroy()
		}
		arcIn.Destroy()
		arcOut.Destroy()
		if sessionOpts != nil {
			sessionOpts.Destroy()
		}
		return nil, fmt.Errorf("arc session create: %w", err)
	}
	// SessionOptions can be destroyed after all sessions are created
	if sessionOpts != nil {
		sessionOpts.Destroy()
	}

	return &sessionPair{
		detSession:   detSession,
		arcSession:   arcSession,
		detInput:     detIn,
		arcInput:     arcIn,
		detOuts:      detGroups,
		arcOutput:    arcOut,
		arcBatchSz:   batchSz,
		concatScores: concatScores,
		concatBboxes: concatBboxes,
		concatLm:     concatLm,
		splitScores:  splitScores,
		splitBboxes:  splitBboxes,
		splitLm:      splitLm,
	}, nil
}

// buildSplitSession creates a detector session with 9 split output tensors.
func buildSplitSession(detPath string, opts ONNXOptions, detIn *ort.Tensor[float32], sessionOpts *ort.SessionOptions) (*ort.AdvancedSession, []ort.Value, []retinaOut, error) {
	const (
		scoresCh = 1 // already after the model's Sigmoid head
		boxesCh  = 4
		landmCh  = 10
	)
	typeChannels := [3]int{scoresCh, boxesCh, landmCh}
	strides := []int{8, 16, 32}
	detOutNames := make([]string, 0, 9)
	detOutputs := make([]ort.Value, 0, 9)
	detGroups := make([]retinaOut, 0, 3)

	for i, s := range strides {
		hw := opts.DetInputSize / s
		ahw := AnchorsPerCell * hw * hw
		grp := retinaOut{stride: s, outputSize: hw * hw}
		for j, name := range opts.DetectorOutputNames[i*3 : i*3+3] {
			c := typeChannels[j]
			shape := ort.NewShape(int64(ahw), int64(c))
			t, terr := ort.NewEmptyTensor[float32](shape)
			if terr != nil {
				for _, done := range detOutputs {
					if v, ok := done.(*ort.Tensor[float32]); ok && v != nil {
						v.Destroy()
					}
				}
				return nil, nil, nil, fmt.Errorf("det output %s: %w", name, terr)
			}
			detOutputs = append(detOutputs, t)
			switch j {
			case 0:
				grp.scores = t
			case 1:
				grp.bboxes = t
			case 2:
				grp.landmarks = t
			}
			detOutNames = append(detOutNames, name)
		}
		detGroups = append(detGroups, grp)
	}

	session, err := ort.NewAdvancedSession(
		detPath,
		[]string{opts.DetectorInputName},
		detOutNames,
		[]ort.Value{detIn},
		detOutputs,
		sessionOpts,
	)
	if err != nil {
		for _, t := range detOutputs {
			if v, ok := t.(*ort.Tensor[float32]); ok && v != nil {
				v.Destroy()
			}
		}
		return nil, nil, nil, fmt.Errorf("det session create: %w", err)
	}
	return session, detOutputs, detGroups, nil
}

// buildConcatSession creates a detector session with 3 concatenated output tensors.
func buildConcatSession(detPath string, opts ONNXOptions, detIn *ort.Tensor[float32], sessionOpts *ort.SessionOptions) (*ort.AdvancedSession, *ort.Tensor[float32], *ort.Tensor[float32], *ort.Tensor[float32], error) {
	totalAnchors := int64(16800) // 12800 + 3200 + 800

	scoresShape := ort.NewShape(1, totalAnchors, 2)
	scores, err := ort.NewEmptyTensor[float32](scoresShape)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("det concat scores: %w", err)
	}

	bboxesShape := ort.NewShape(1, totalAnchors, 4)
	bboxes, err := ort.NewEmptyTensor[float32](bboxesShape)
	if err != nil {
		scores.Destroy()
		return nil, nil, nil, nil, fmt.Errorf("det concat bboxes: %w", err)
	}

	lmShape := ort.NewShape(1, totalAnchors, 10)
	lm, err := ort.NewEmptyTensor[float32](lmShape)
	if err != nil {
		scores.Destroy()
		bboxes.Destroy()
		return nil, nil, nil, nil, fmt.Errorf("det concat landmarks: %w", err)
	}

	detOutputs := []ort.Value{bboxes, scores, lm} // matches bbox, confidence, landmark order

	session, err := ort.NewAdvancedSession(
		detPath,
		[]string{opts.DetectorInputName},
		opts.DetectorOutputNames,
		[]ort.Value{detIn},
		detOutputs,
		sessionOpts,
	)
	if err != nil {
		scores.Destroy()
		bboxes.Destroy()
		lm.Destroy()
		return nil, nil, nil, nil, fmt.Errorf("det session create: %w", err)
	}
	return session, scores, bboxes, lm, nil
}

// makeSplitGroups creates 3 empty retinaOut groups for holding split concat data.
func makeSplitGroups(detInputSize int) []retinaOut {
	strides := []int{8, 16, 32}
	groups := make([]retinaOut, 3)
	for i, s := range strides {
		hw := detInputSize / s
		groups[i] = retinaOut{stride: s, outputSize: hw * hw}
	}
	return groups
}

// splitConcatOutputs copies data from 3 concat tensors into the 9 per-stride
// tensors in detOuts. Called after detSession.Run() for concat-format detectors.
// Uses pre-allocated split tensors to avoid per-call allocations.
func splitConcatOutputs(sp *sessionPair, opts ONNXOptions) {
	if sp.concatScores == nil || sp.concatBboxes == nil || sp.concatLm == nil {
		return
	}

	scoresData := sp.concatScores.GetData()
	bboxesData := sp.concatBboxes.GetData()
	lmData := sp.concatLm.GetData()

	strides := []int{8, 16, 32}
	anchorOffset := 0

	for i, s := range strides {
		hw := opts.DetInputSize / s
		ahw := AnchorsPerCell * hw * hw
		outputSize := hw * hw

		// Reuse pre-allocated tensors
		scores := sp.splitScores[i]
		bboxes := sp.splitBboxes[i]
		lm := sp.splitLm[i]

		scoresBuf := scores.GetData()
		bboxesBuf := bboxes.GetData()
		lmBuf := lm.GetData()

		for j := 0; j < ahw; j++ {
			// Extract face channel (index 1) from 2-channel confidence
			srcIdx := anchorOffset + j
			scoresBuf[j] = scoresData[srcIdx*2+1] // face channel

			// Copy bbox and landmarks directly
			copy(bboxesBuf[j*4:(j+1)*4], bboxesData[srcIdx*4:(srcIdx+1)*4])
			copy(lmBuf[j*10:(j+1)*10], lmData[srcIdx*10:(srcIdx+1)*10])
		}

		sp.detOuts[i].scores = scores
		sp.detOuts[i].bboxes = bboxes
		sp.detOuts[i].landmarks = lm
		sp.detOuts[i].outputSize = outputSize

		anchorOffset += ahw
	}
}

func destroyDetector(s *ort.AdvancedSession, in *ort.Tensor[float32], outs []ort.Value) {
	if s != nil {
		s.Destroy()
	}
	if in != nil {
		in.Destroy()
	}
	for _, v := range outs {
		if t, ok := v.(*ort.Tensor[float32]); ok && t != nil {
			t.Destroy()
		}
	}
}

func destroySessionPair(sp *sessionPair) {
	if sp == nil {
		return
	}
	if sp.detSession != nil {
		sp.detSession.Destroy()
	}
	if sp.arcSession != nil {
		sp.arcSession.Destroy()
	}
	if sp.detInput != nil {
		sp.detInput.Destroy()
	}
	if sp.arcInput != nil {
		sp.arcInput.Destroy()
	}
	if sp.arcOutput != nil {
		sp.arcOutput.Destroy()
	}
	// Destroy concat tensors if present
	if sp.concatScores != nil {
		sp.concatScores.Destroy()
	}
	if sp.concatBboxes != nil {
		sp.concatBboxes.Destroy()
	}
	if sp.concatLm != nil {
		sp.concatLm.Destroy()
	}
	// Destroy per-stride split tensors
	for i := range sp.detOuts {
		if sp.detOuts[i].scores != nil {
			sp.detOuts[i].scores.Destroy()
		}
		if sp.detOuts[i].bboxes != nil {
			sp.detOuts[i].bboxes.Destroy()
		}
		if sp.detOuts[i].landmarks != nil {
			sp.detOuts[i].landmarks.Destroy()
		}
	}
}

// Extract inspects img, returns one L2-normalized 512-d vector per
// detected face. Safe for concurrent use.
//
// Implemented as a thin wrapper over ExtractWithDebug with no sink;
// the --debug path is canonical and Extract is the convenience for
// callers that don't care about per-face metadata.
func (e *ONNXEmbedder) Extract(ctx context.Context, img image.Image) ([][]float32, error) {
	return e.ExtractWithDebug(ctx, img, "", nil)
}

// ExtractFile reads from disk and calls Extract.
func (e *ONNXEmbedder) ExtractFile(ctx context.Context, path string) ([][]float32, error) {
	img, err := imaging.Open(path, imaging.AutoOrientation(true))
	if err != nil {
		return nil, err
	}
	return e.Extract(ctx, img)
}

// ExtractWithDebug is the --debug variant. It returns the same
// embeddings as Extract and additionally notifies sink for every
// successfully aligned face. sink may be nil; a nil sink makes the
// call equivalent to Extract.
//
// The 112×112 Aligned image is the SAME pointer fed to ArcFace (we
// re-use it to avoid a second affine warp), so a sink holding it
// will see exactly what the recognizer saw. The Embedding handed to
// the sink is the freshly L2-normalized vector for that face — sinks
// MUST treat it as read-only (its backing may be shared with the
// returned slice; see FileDebugSink for the canonical sink).
//
// srcPath is propagated verbatim to sink.OnFace. extractWith does
// not look at the path — it just threads it through.
func (e *ONNXEmbedder) ExtractWithDebug(ctx context.Context, img image.Image, srcPath string, sink DebugSink) ([][]float32, error) {
	if img == nil {
		return nil, errors.New("onnx: nil image")
	}
	if e.pool == nil {
		return nil, errors.New("onnx: embedder not initialized")
	}
	select {
	case sp := <-e.pool:
		defer func() { e.pool <- sp }()
		return e.extractWithDebug(ctx, img, srcPath, sp, sink)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Close releases all ONNX resources. Idempotent.
func (e *ONNXEmbedder) Close() error {
	var firstErr error
	e.closeOnce.Do(func() {
		e.closePoolAndDestroy()
		if err := ort.DestroyEnvironment(); err != nil && firstErr == nil {
			firstErr = err
		}
	})
	return firstErr
}

// extractWithDebug runs the per-image workload end-to-end on one
// worker. sink may be nil; when non-nil, it is called once per
// successfully aligned face with the freshly computed 112×112
// ArcFace-aligned crop (the same pointer fed to ArcFace), bbox in
// the source image's coordinate system, and the L2-normalized
// embedding. The sink's Face.Embedding points to the same backing
// array as the corresponding element of the returned slice; sinks
// MUST treat it as read-only.
func (e *ONNXEmbedder) extractWithDebug(ctx context.Context, img image.Image, srcPath string, sp *sessionPair, sink DebugSink) ([][]float32, error) {
	// 1. preprocess detector input.
	if err := preprocessRetinaFace(img, e.detInputSz, sp.detInput); err != nil {
		return nil, fmt.Errorf("retina preprocess: %w", err)
	}
	// 2. detect.
	if err := sp.detSession.Run(); err != nil {
		return nil, fmt.Errorf("retina run: %w", err)
	}
	// For concat-format detectors, split outputs into per-stride groups
	if sp.concatScores != nil {
		splitConcatOutputs(sp, e.opts)
	}
	// Debug: log top-10 scoring anchors before threshold/NMS.
	// Gated: building+sorting 16800 anchors is real CPU (reflection-
	// based sort.Slice) and real allocation on EVERY image, including
	// the non---debug Extract() path (which routes through this same
	// function). Only pay for it when someone can actually see the
	// output.
	if slog.Default().Enabled(ctx, slog.LevelDebug) {
		logTopNCandidates(sp.detOuts, img.Bounds(), 10)
	}
	// 3. decode + NMS.
	faces := decodeRetinaFaces(sp.detOuts, img.Bounds(), e.opts.DetThreshold, e.opts.NMSIoU, e.opts.LandmarkVarianceBaked)
	if len(faces) == 0 {
		return [][]float32{}, nil
	}
	// 4. align all faces first (collect aligned images).
	type alignedFace struct {
		aligned *image.RGBA
		face    face
		index   int
	}
	var alignedFaces []alignedFace
	for i, f := range faces {
		slog.Default().Debug("face pre-align",
			"i", i,
			"score", f.Score,
			"bbox", fmt.Sprintf("(%g,%g)->(%g,%g)", f.X1, f.Y1, f.X2, f.Y2),
			"lm0", fmt.Sprintf("(%g,%g)", f.Landmarks[0][0], f.Landmarks[0][1]),
			"lm1", fmt.Sprintf("(%g,%g)", f.Landmarks[1][0], f.Landmarks[1][1]),
			"lm2", fmt.Sprintf("(%g,%g)", f.Landmarks[2][0], f.Landmarks[2][1]),
			"lm3", fmt.Sprintf("(%g,%g)", f.Landmarks[3][0], f.Landmarks[3][1]),
			"lm4", fmt.Sprintf("(%g,%g)", f.Landmarks[4][0], f.Landmarks[4][1]),
		)
		aligned, err := alignFaceFromBox(img, f)
		if err != nil {
			slog.Default().Debug("align failed, skipping face",
				"i", i, "err", err, "score", f.Score)
			continue
		}
		alignedFaces = append(alignedFaces, alignedFace{aligned: aligned, face: f, index: i})
	}
	if len(alignedFaces) == 0 {
		return [][]float32{}, nil
	}
	// 5. batch-recognize aligned faces.
	out := make([][]float32, 0, len(alignedFaces))
	batchSz := sp.arcBatchSz
	if batchSz < 1 {
		batchSz = 1
	}
	for start := 0; start < len(alignedFaces); start += batchSz {
		end := start + batchSz
		if end > len(alignedFaces) {
			end = len(alignedFaces)
		}
		batch := alignedFaces[start:end]
		// Write batch to tensor
		imgs := make([]*image.RGBA, len(batch))
		for i, af := range batch {
			imgs[i] = af.aligned
		}
		n := writeAlignedBatchToTensor(imgs, sp.arcInput)
		if n == 0 {
			continue
		}
		// Run ArcFace
		if err := sp.arcSession.Run(); err != nil {
			return nil, fmt.Errorf("arcface run: %w", err)
		}
		// Copy outputs
		outData := sp.arcOutput.GetData()
		for i := 0; i < n; i++ {
			start := i * ArcFaceDim
			end := start + ArcFaceDim
			v := append([]float32(nil), outData[start:end]...)
			l2Normalize(v)
			out = append(out, v)
			if sink != nil {
				sink.OnFace(ctx, srcPath, Face{
					Embedding: v,
					BBox:      image.Rect(int(batch[i].face.X1), int(batch[i].face.Y1), int(batch[i].face.X2), int(batch[i].face.Y2)),
					Aligned:   batch[i].aligned,
					Score:     batch[i].face.Score,
					Index:     batch[i].index,
				})
			}
		}
	}
	return out, nil
}

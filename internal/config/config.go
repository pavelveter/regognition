// Package config merges INI file settings with command-line flags.
//
// Precedence (highest to lowest):
//  1. CLI flag explicitly set (non-zero / non-empty)
//  2. INI value (auto-searched next to the binary, then in CWD)
//  3. Built-in defaults
//
// INI keys must live under one of [paths], [ml], or [pipeline] sections.
// (Flat root keys were once supported as a fallback but produced
// surprising results when a section value equalled the default.)
package config

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/ini.v1"
)

// Config holds resolved runtime settings.
type Config struct {
	// Paths
	PersonaDir string // ini: paths.persona_dir | flag: --persona
	SearchDir  string // ini: paths.dir_search  | flag: --search
	OutputDir  string // ini: paths.dir_finded  | flag: --out
	DirSkip    string // ini: paths.dir_skip    | flag: --dir-skip (comma-separated folder names to skip)
	SkipFiles  string // ini: paths.skip_files  | flag: --skip-files (glob patterns for files to skip, comma-separated)
	Extensions string // ini: paths.extensions  | flag: --extensions (allowed extensions, comma-separated, e.g. ".jpg,.jpeg,.png")
	ConfigPath string // flag: --config (absolute path; resolved by Load)

	// Models
	DetectorModelPath  string // ini: ml.detector_model
	EmbedderModelPath  string // ini: ml.embedder_model
	OrtLibPath         string // ini: ml.ort_lib         | flag: --ort-lib
	DetectorModelURL   string // ini: ml.detector_model_url | flag: --detector-url
	EmbedderModelURL   string // ini: ml.embedder_model_url | flag: --embedder-url
	AutoDownload       bool   // ini: ml.auto_download   | flag: --no-download (negation)
	EmbedderOutputName string // ini: ml.embedder_output | flag: --embedder-output
	EmbedderInputName  string // ini: ml.embedder_input  | flag: --embedder-input
	DetectorScoresName string // ini: ml.detector_scores | flag: --detector-scores
	DetectorBoxesName  string // ini: ml.detector_boxes  | flag: --detector-boxes
	DetectorLandmName  string // ini: ml.detector_landm  | flag: --detector-landm
	// Detector format: "split" (9 tensors for MobileNet) or
	// "concat" (3 tensors for ResNet50). Auto-detected if empty.
	DetectorFormat string // ini: ml.detector_format
	// Whether the ONNX export bakes variance[0] into landmark outputs.
	// MobileNet: true, ResNet50: false.
	LandmarkVarianceBaked *bool // ini: ml.landmark_variance_baked
	// Detector input tensor name. MobileNet: "input.1", ResNet50: "input".
	DetectorInputName string // ini: ml.detector_input
	// Execution Providers
	CoreML bool // ini: ml.coreml | flag: --coreml
	// Batching
	ArcBatchSize int // ini: ml.arc_batch_size | flag: --arc-batch-size

	// Pipeline tuning
	Workers         int     // ini: pipeline.workers         | flag: --workers
	IOWorkers       int     // ini: pipeline.io_workers      — concurrent file reads (0 = default 4)
	TargetDimension int     // ini: pipeline.target_dimension (max-edge cap; aspect ratio preserved)
	Threshold       float64 // ini: pipeline.threshold       | flag: --threshold
	DetInputSize    int     // ini: pipeline.det_input_size
	DetThreshold    float64 // ini: pipeline.det_threshold
	NMSIoU          float64 // ini: pipeline.nms_iou

	// Cache (optional)
	CachePath string // ini: pipeline.cache_path (empty -> NoopCache)

	// UI / logging
	LogLevel string // ini: ui.log_level | flag: --log-level | default "debug"
	Color    string // ini: ui.color     | flag: --color     | default "auto"
	Debug    bool   // ini: ui.debug     | flag: --debug     | default false
}

// boolPtr returns a pointer to b. Used for *bool config defaults.
func boolPtr(b bool) *bool { return &b }

// Defaults returns Config with built-in defaults for every field.
func Defaults() Config {
	return Config{
		PersonaDir: "persona",
		SearchDir:  "./archive",
		OutputDir:  "./finded",
		Workers:    8,
		// Max-edge cap for archive photos: portrait frames will be 1280 tall,
		// landscape frames 1280 wide — aspect ratio is preserved by
		// imaging.Fit inside pipeline.processJob.
		TargetDimension: 1280,
		Threshold:       0.55,
		DetInputSize:    640,
		DetThreshold:    0.5,
		NMSIoU:          0.4,
		// Biubug6 Pytorch_Retinaface exports emit the canvas as the
		// model's second graph input — PyTorch's ONNX exporter calls it
		// "input.1" (the first slot usually holds the placeholder).
		EmbedderInputName: "input.1",
		// arcface_r50 export emits a single 512-d vector named "516"
		// in declaration order. Override via INI / --embedder-output.
		EmbedderOutputName: "516",
		DetectorScoresName: "cls",
		DetectorBoxesName:  "bbox",
		DetectorLandmName:  "landm",
		// MobileNet (the shipped retinaface_mnet025_v2 default)
		// bakes variance[0] into landmark outputs; only ResNet50
		// exports need this false. Override via
		// [ml] landmark_variance_baked = false if you swap models.
		LandmarkVarianceBaked: boolPtr(false),

		// UI/logging defaults: emit everything by default ("debug" so
		// the operator sees per-image activity during long runs) and let
		// prettylog decide whether to colour by auto-detecting TTY.
		// Debug is opt-in via --debug / [ui] debug = true; when set, it
		// also turns on face-crop saving and bypasses the cache.
		LogLevel: "info",
		Color:    "auto",
		Debug:    false,

		// Default auto-download source: HuggingFace mirror of insightface's
		// buffalo_s pack (det_500m.onnx + w600k_mbf.onnx, IR version 6).
		// The bytes are saved under cfg.DetectorModelPath / EmbedderModelPath.
		DetectorModelURL: "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/det_500m.onnx",
		EmbedderModelURL: "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx",
		AutoDownload:     true,
	}
}

// Load registers CLI flags on fs, parses arguments, locates the INI
// file (binary dir → CWD when --config was not passed), merges INI,
// applies CLI overrides, and validates the configuration.
//
// Search policy for `config.ini` when --config is omitted:
//  1. <dir-of-running-binary>/config.ini
//  2. <cwd>/config.ini
//  3. error with a hint pointing the operator at --config.
func Load(fs *flag.FlagSet, args []string) (*Config, error) {
	cfg := Defaults()

	var (
		cliConfigPath     = fs.String("config", "", "explicit INI file path (defaults to auto-search for config.ini next to the binary, then CWD)")
		cliPersona        = fs.String("persona", "", "persona selfies directory")
		cliSearch         = fs.String("search", "", "directory to scan for matches")
		cliOutput         = fs.String("out", "", "directory to copy matched photos to")
		cliDirSkip        = fs.String("dir-skip", "", "comma-separated folder names to skip during scan")
		cliSkipFiles      = fs.String("skip-files", "", "glob patterns for files to skip (comma-separated, e.g. \"._*,*.tmp\")")
		cliExtensions     = fs.String("extensions", "", "allowed file extensions (comma-separated, e.g. \".jpg,.jpeg,.png\"); empty = default set")
		cliWorkers        = fs.Int("workers", 0, "number of pipeline workers")
		cliThreshold      = fs.Float64("threshold", -1, "cosine distance threshold (matches are <= this)")
		cliCache          = fs.String("cache", "", "sqlite cache path (empty disables cache)")
		cliOrtLib         = fs.String("ort-lib", "", "path to libonnxruntime shared library (.dylib/.so/.dll)")
		cliNoDownload     = fs.Bool("no-download", false, "disable auto-download of missing ONNX models on startup")
		cliDetectorURL    = fs.String("detector-url", "", "URL the auto-downloader fetches the detector ONNX from")
		cliEmbedderURL    = fs.String("embedder-url", "", "URL the auto-downloader fetches the recognizer ONNX from")
		cliEmbedderOutput = fs.String("embedder-output", "", "name of recognizer output tensor")
		cliEmbedderInput  = fs.String("embedder-input", "", "name of recognizer input tensor")
		cliDetectorScores = fs.String("detector-scores", "", "suffix per-stride for detector score outputs")
		cliDetectorBoxes  = fs.String("detector-boxes", "", "suffix per-stride for detector bbox outputs")
		cliDetectorLandm  = fs.String("detector-landm", "", "suffix per-stride for detector landmark outputs")
		cliLogLevel       = fs.String("log-level", "", "minimum log level: debug|info|warn|error (default 'debug'; 'error' = quiet)")
		cliColor          = fs.String("color", "", "ANSI color usage: auto|always|never (default 'auto')")
		cliDebug          = fs.Bool("debug", false, "debug mode: force DEBUG log level AND save 112×112 ArcFace-aligned face crops for every match into <out>/debug/, mirroring the search dir structure. Bypasses the embedding cache. Off by default.")
		cliCoreML         = fs.Bool("coreml", false, "enable CoreML acceleration on macOS (Apple Neural Engine / GPU / CPU)")
		cliArcBatchSize   = fs.Int("arc-batch-size", 0, "max faces per ArcFace inference call (0 = no batching)")
	)
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	// 1. Locate and load INI.
	iniPath, err := locateConfig(*cliConfigPath)
	if err != nil {
		return nil, err
	}
	if iniPath != "" {
		iniFile, err := ini.Load(iniPath)
		if err != nil {
			return nil, fmt.Errorf("load ini %q: %w", iniPath, err)
		}
		mergeINI(&cfg, iniFile)
		cfg.ConfigPath = iniPath
	}

	// 2. CLI overrides.
	if *cliPersona != "" {
		cfg.PersonaDir = *cliPersona
	}
	if *cliSearch != "" {
		cfg.SearchDir = *cliSearch
	}
	if *cliOutput != "" {
		cfg.OutputDir = *cliOutput
	}
	if *cliDirSkip != "" {
		cfg.DirSkip = *cliDirSkip
	}
	if *cliSkipFiles != "" {
		cfg.SkipFiles = *cliSkipFiles
	}
	if *cliExtensions != "" {
		cfg.Extensions = *cliExtensions
	}
	if *cliWorkers > 0 {
		cfg.Workers = *cliWorkers
	}
	if *cliThreshold >= 0 {
		cfg.Threshold = *cliThreshold
	}
	if *cliCache != "" {
		cfg.CachePath = *cliCache
	}
	if *cliOrtLib != "" {
		cfg.OrtLibPath = *cliOrtLib
	}
	if *cliNoDownload {
		cfg.AutoDownload = false
	}
	if *cliDetectorURL != "" {
		cfg.DetectorModelURL = *cliDetectorURL
	}
	if *cliEmbedderURL != "" {
		cfg.EmbedderModelURL = *cliEmbedderURL
	}
	if *cliEmbedderOutput != "" {
		cfg.EmbedderOutputName = *cliEmbedderOutput
	}
	if *cliEmbedderInput != "" {
		cfg.EmbedderInputName = *cliEmbedderInput
	}
	if *cliDetectorScores != "" {
		cfg.DetectorScoresName = *cliDetectorScores
	}
	if *cliDetectorBoxes != "" {
		cfg.DetectorBoxesName = *cliDetectorBoxes
	}
	if *cliDetectorLandm != "" {
		cfg.DetectorLandmName = *cliDetectorLandm
	}
	if *cliLogLevel != "" {
		cfg.LogLevel = *cliLogLevel
	}
	if *cliColor != "" {
		cfg.Color = *cliColor
	}
	if *cliDebug {
		cfg.Debug = true
	}
	if *cliCoreML {
		cfg.CoreML = true
	}
	if *cliArcBatchSize > 0 {
		cfg.ArcBatchSize = *cliArcBatchSize
	}

	// 3. Validation.
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// locateConfig decides where to find the INI file.
//
//   - explicitPath != ""  → operator passed --config; the file MUST
//     exist at that exact path. Returns an error if it does not.
//   - explicitPath == ""  → auto-search for `config.ini` next to the
//     binary, then in CWD. Returns an error if neither has the file.
func locateConfig(explicitPath string) (string, error) {
	if explicitPath != "" {
		// Honour --config strictly: relative paths resolve against CWD.
		abs, err := filepath.Abs(explicitPath)
		if err != nil {
			return "", fmt.Errorf("--config %q: %w", explicitPath, err)
		}
		if _, err := os.Stat(abs); err != nil {
			return "", fmt.Errorf("--config %q not found (resolved: %q): %w", explicitPath, abs, err)
		}
		return abs, nil
	}

	const defaultName = "config.ini"
	var tried []string
	for _, dir := range configSearchDirs() {
		candidate := filepath.Join(dir, defaultName)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
		tried = append(tried, candidate)
	}
	return "", fmt.Errorf("config.ini not found anywhere; tried: %v (use --config /path/to/file.ini)", tried)
}

// configSearchDirs lists the directories to scan when --config is
// omitted. Order is: binary's own directory, then CWD. Centralised
// here so future additions (e.g. $XDG_CONFIG_HOME/regognition) have
// one place to live.
func configSearchDirs() []string {
	dirs := make([]string, 0, 2)
	if exe, err := os.Executable(); err == nil {
		dirs = append(dirs, filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, cwd)
	}
	return dirs
}

// mergeINI reads keys from [paths], [ml], [pipeline] sections. Root-level
// (flat) keys are intentionally NOT honored — see package doc.
func mergeINI(cfg *Config, f *ini.File) {
	// [paths]
	s := f.Section("paths")
	if v := s.Key("persona_dir").String(); v != "" {
		cfg.PersonaDir = v
	}
	if v := s.Key("dir_search").String(); v != "" {
		cfg.SearchDir = v
	}
	if v := s.Key("dir_finded").String(); v != "" {
		cfg.OutputDir = v
	}
	if v := s.Key("dir_skip").String(); v != "" {
		cfg.DirSkip = v
	}
	if v := s.Key("skip_files").String(); v != "" {
		cfg.SkipFiles = v
	}
	if v := s.Key("extensions").String(); v != "" {
		cfg.Extensions = v
	}

	// [ml]
	s = f.Section("ml")
	if v := s.Key("detector_model").String(); v != "" {
		cfg.DetectorModelPath = v
	}
	if v := s.Key("embedder_model").String(); v != "" {
		cfg.EmbedderModelPath = v
	}
	if v := s.Key("ort_lib").String(); v != "" {
		cfg.OrtLibPath = v
	}
	if v, err := s.Key("auto_download").Bool(); err == nil {
		cfg.AutoDownload = v
	}
	if v := s.Key("detector_model_url").String(); v != "" {
		cfg.DetectorModelURL = v
	}
	if v := s.Key("embedder_model_url").String(); v != "" {
		cfg.EmbedderModelURL = v
	}
	if v := s.Key("embedder_output").String(); v != "" {
		cfg.EmbedderOutputName = v
	}
	if v := s.Key("embedder_input").String(); v != "" {
		cfg.EmbedderInputName = v
	}
	if v := s.Key("detector_scores").String(); v != "" {
		cfg.DetectorScoresName = v
	}
	if v := s.Key("detector_boxes").String(); v != "" {
		cfg.DetectorBoxesName = v
	}
	if v := s.Key("detector_landm").String(); v != "" {
		cfg.DetectorLandmName = v
	}
	if v := s.Key("detector_format").String(); v != "" {
		cfg.DetectorFormat = v
	}
	if v, err := s.Key("landmark_variance_baked").Bool(); err == nil {
		cfg.LandmarkVarianceBaked = &v
	}
	if v := s.Key("detector_input").String(); v != "" {
		cfg.DetectorInputName = v
	}
	if v, err := s.Key("coreml").Bool(); err == nil {
		cfg.CoreML = v
	}
	if v, err := s.Key("arc_batch_size").Int(); err == nil && v > 0 {
		cfg.ArcBatchSize = v
	}

	// [pipeline]
	s = f.Section("pipeline")
	if v, err := s.Key("workers").Int(); err == nil && v > 0 {
		cfg.Workers = v
	}
	if v, err := s.Key("io_workers").Int(); err == nil && v > 0 {
		cfg.IOWorkers = v
	}
	if v, err := s.Key("target_dimension").Int(); err == nil && v > 0 {
		cfg.TargetDimension = v
	}
	if v, err := s.Key("threshold").Float64(); err == nil && v > 0 {
		cfg.Threshold = v
	}
	if v := s.Key("cache_path").String(); v != "" {
		cfg.CachePath = v
	}
	if v, err := s.Key("det_input_size").Int(); err == nil && v > 0 {
		cfg.DetInputSize = v
	}
	if v, err := s.Key("det_threshold").Float64(); err == nil && v > 0 {
		cfg.DetThreshold = v
	}
	if v, err := s.Key("nms_iou").Float64(); err == nil && v > 0 {
		cfg.NMSIoU = v
	}

	// [ui]
	s = f.Section("ui")
	if v := s.Key("log_level").String(); v != "" {
		cfg.LogLevel = v
	}
	if v := s.Key("color").String(); v != "" {
		cfg.Color = v
	}
	if v, err := s.Key("debug").Bool(); err == nil {
		cfg.Debug = v
	}
}

// Validate asserts required fields and ranges.
func (c Config) Validate() error {
	if c.PersonaDir == "" {
		return errors.New("persona_dir is required")
	}
	if c.SearchDir == "" {
		return errors.New("dir_search is required")
	}
	if c.OutputDir == "" {
		return errors.New("dir_finded is required")
	}
	if c.Workers <= 0 {
		return errors.New("workers must be positive")
	}
	if c.TargetDimension <= 0 {
		return errors.New("target_dimension must be positive")
	}
	if c.Threshold <= 0 || c.Threshold > 2 {
		return errors.New("threshold must be in (0, 2] (cosine distance)")
	}
	if c.DetInputSize <= 0 {
		return errors.New("det_input_size must be positive")
	}
	if c.DetThreshold <= 0 || c.DetThreshold > 1 {
		return errors.New("det_threshold must be in (0, 1]")
	}
	if c.NMSIoU <= 0 || c.NMSIoU > 1 {
		return errors.New("nms_iou must be in (0, 1]")
	}
	return nil
}

// Summary is a redacted single-line representation for logs.
func (c Config) Summary() string {
	return fmt.Sprintf(
		"persona=%q search=%q out=%q workers=%d threshold=%.3f max_edge=%d det=%d@%.2f nms=%.2f embedder=%s/%s detector=%s/%s/%s",
		c.PersonaDir, c.SearchDir, c.OutputDir, c.Workers, c.Threshold,
		c.TargetDimension, c.DetInputSize, c.DetThreshold, c.NMSIoU,
		c.EmbedderInputName, c.EmbedderOutputName,
		c.DetectorScoresName, c.DetectorBoxesName, c.DetectorLandmName,
	)
}

# regognition

A fast Go CLI utility that finds photos of a specific person inside a
local photo archive. Given 1+ selfies in `persona/` and a photo archive
in `dir_search`, it copies photos whose faces match the target person
into `dir_finded`.

Built on **RetinaFace** (face detection) + **ArcFace** (embedding)
via ONNX Runtime with CoreML acceleration on macOS. Includes a SQLite
embedding cache and a prefetch I/O pipeline optimized for slow storage.

## Features

- **RetinaFace ResNet50** detector — high-precision face detection with NMS
- **ArcFace w600k_r50** embedder — 512-d L2-normalized vectors (MR-ALL 91.25)
- **CoreML EP** — Apple Neural Engine / GPU / CPU fallback on macOS
- **SQLite embedding cache** — repeat runs skip inference for unchanged files
- **Prefetch I/O pipeline** — sequential reads + parallel decode, ideal for USB/network drives
- **Directory skip** — exclude folders by name during scan (`dir_skip`)
- **Auto-download** — fetches ONNX models from HuggingFace on first run
- **Debug mode** — saves aligned face crops + per-face distance logs (`--debug`)

## Project layout

```
regognition/
├ main.go                      # entry point, ties everything together
├ config.ini.example           # sample INI (copy to ./config.ini)
├ go.mod
├ models/                      # ONNX weights (auto-downloaded on first run)
└ internal/
    ├── assetfetch/             # HTTP model downloader with progress logging
    ├── cache/                  # SQLite embedding cache + CachedEmbedder
    ├── config/                 # INI + flag loading
    ├── cosine/                 # cosine distance math
    ├── embedder/               # Embedder interface + ONNX + stub impl
    ├── persona/                # persona selfies + embedding extraction
    ├── pipeline/               # prefetch I/O pipeline + worker pool
    ├── prettylog/              # colored slog handler
    └── scanner/                # recursive image walker
```

## Requirements

- Go 1.21+
- `libonnxruntime.dylib` (macOS) or `.so` (Linux) — auto-detected from `[ml].ort_lib`
- ONNX models: `retinaface_resnet50.onnx` + `w600k_r50.onnx` (auto-downloaded if missing)

## Build & run

```bash
go mod tidy
go build -o regognition .

# Prepare directories (place 1+ selfies in persona/, photos in archive/):
mkdir -p persona archive matched

# Copy and edit config:
cp config.ini.example config.ini

./regognition
```

CLI flags always override the INI:

```bash
./regognition \
  --persona ./selfies \
  --search  ./photos_in \
  --out     ./photos_out \
  --workers 8 \
  --threshold 0.65 \
  --debug
```

`--help` lists every flag.

## Configuration

Copy `config.ini.example` to `config.ini` and adjust paths:

```ini
[paths]
persona_dir = ./persona
dir_search  = ./archive
dir_finded  = ./matched
dir_skip    = "2. Для печати и дизайна"

[ml]
detector_model = ./models/retinaface_resnet50.onnx
embedder_model = ./models/w600k_r50.onnx
ort_lib        = ./models/libonnxruntime.dylib
coreml         = true
detector_format = concat

[pipeline]
workers          = 8
target_dimension = 1280
threshold        = 0.65
det_input_size   = 640
det_threshold    = 0.5
nms_iou          = 0.4
cache_path       = ./cache.sqlite
```

## Models

| Model | Path | Source | Size | Notes |
|-------|------|--------|------|-------|
| Detector | `models/retinaface_resnet50.onnx` | insightface buffalo_l | ~109 MB | ResNet50, 3-tensor concat format |
| Embedder | `models/w600k_r50.onnx` | insightface buffalo_l | ~166 MB | ArcFace 512-d, fixed batch=1 |

Auto-download fetches from HuggingFace by default. Use `--no-download`
for air-gapped environments.

### Detector formats

- `concat` (default) — 3 tensors for ResNet50 (scores/bbox/lmk per stride)
- `split` — 9 tensors for MobileNet variants

### CoreML

Set `coreml = true` in `[ml]` to enable Apple Neural Engine / GPU
acceleration. First run compiles the CoreML model (~45s), subsequent
runs are fast.

## Performance

- **Cold run**: ~4–5 img/sec (NVMe, CoreML)
- **Warm cache**: ~5 img/sec (inference skipped for cached files)
- **Slow storage**: prefetch pipeline batches reads + parallel decode

The prefetch pipeline reads files in configurable batches (default 4)
and decodes images in parallel goroutines, keeping workers focused on
CPU-bound inference. This is critical for USB/network drives where
random I/O kills throughput.

## Embedding cache

SQLite-backed cache at `cache.sqlite` (configurable via `cache_path`).
On repeat runs, files are looked up by path + hash (mtime+size).
Unchanged files skip RetinaFace+ArcFace entirely — only cosine scoring
runs. Files with no detected faces are cached too (dominant cost on
real archives).

Delete `cache.sqlite` to rebuild. Schema:

```sql
CREATE TABLE IF NOT EXISTS photo_cache (
    path        TEXT PRIMARY KEY,
    hash        TEXT NOT NULL,
    faces_count INTEGER NOT NULL,
    embeddings  BLOB NOT NULL
);
```

## Debug mode

`--debug` saves aligned 112×112 face crops under `<output>/debug/` and
logs per-face detector scores, landmarks, and cosine distances. The
embedding cache is bypassed in debug mode (fresh inference needed for
crop extraction).

## License

This project is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for the full text.

# regognition

A small Go CLI utility that finds photos of a specific person inside a
local photo archive. Given 1+ selfies in `persona/` and a photo archive in
`dir_search`, it copies photos whose faces match the target person into
`dir_finded`.

This repository is a **skeleton**: the ML pipeline works end-to-end on a
deterministic stub embedder so the rest of the project (config, scanner,
pipeline, CLI) can be exercised and validated. Replace the stub with a
RetinaFace + ArcFace ONNX implementation when ready.

## Project layout

```
regognition/
├── main.go                      # entry point, ties everything together
├── config.ini.example           # sample INI (copy to ./config.ini)
├── go.mod
├── models/                      # drop ONNX files here (currently empty)
├── TODO.md                      # original blueprint
└── internal/
    ├── config/                  # INI + flag loading
    ├── scanner/                 # recursive image walker
    ├── persona/                 # persona selfies + embeddings
    ├── embedder/                # ML abstraction + deterministic stub
    ├── cosine/                  # cosine distance math
    └── pipeline/                # channel-based worker pool
```

## Build & run

```bash
go mod tidy
go build ./...

# Prepare directories (place 1+ selfies in persona/, photos in archive/):
mkdir -p persona archive finded

# Run with default config.ini (or copy from the example):
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
  --threshold 0.40
```

`--help` lists every flag.

## How the stub works (and how to test the pipeline end-to-end)

`internal/embedder/stub.go` returns one 512-d L2-normalized vector per
image, seeded from the central pixel + a deterministic LCG over a
sampled pixel subset + (optionally) the file path. Identical inputs
produce identical vectors, so:

1. Drop a single selfie into `persona/`.
2. Drop the **same** file into `archive/`.
3. Run `./regognition`.

The stub will report cosine distance ≈ 0.0 and the file will be copied to
`finded/`. Two different images will produce non-trivial distance,
letting you sanity-check the threshold.

## Swapping in real ML

The pipeline only depends on the `embedder.Embedder` interface:

```go
type Embedder interface {
    Extract(ctx context.Context, img image.Image) ([][]float32, error)
    ExtractFile(ctx context.Context, path string) ([][]float32, error)
    Close() error
}
```

An ONNX-backed implementation lives in `internal/embedder/onnx.go`. It
is selected automatically by `main.go::buildEmbedder` when
`[ml].ort_lib` points at a working `libonnxruntime.{dylib,so,dll}`
**and** both detector/embedder model files exist. Otherwise the stub
runs (with a warning so the fallback is never silent).

### Choosing models

The shipped pair is the **insightface `buffalo_sc` mobile pack** —
fast on Apple Silicon and accurate enough for home-photo archives:

| Path | Source model | What it actually is |
| --- | --- | --- |
| `models/retinaface_mnet025_v2.onnx` | `det_500m.onnx` from buffalo_sc | RetinaFace MobileNet v0.25 (~5 MB) |
| `models/arcface_r50_fp16.onnx`     | `w600k_mbf.onnx` from buffalo_sc | **MobileFaceNet**, NOT ResNet-50 (~13 MB) |
| `models/libonnxruntime.dylib`     | microsoft/onnxruntime v1.26.0 | CPU-only build (CoreML EP available, not enabled) |

If you specifically need **ResNet-50 ArcFace** (heavier but more accurate),
install the `insightface` python package (`pip install insightface` after
creating a venv to bypass PEP 668), then re-run `FaceAnalysis(name="buffalo_l")`.
Drop the `w600k_r50.onnx` it produces into `models/arcface_r50_fp16.onnx`,
and add the matching detector (`scrfd_10g_bnkps.onnx` or similar). The
decoder/alignment code in `onnx_runtime.go` is shape-agnostic — only the
ONNX options (output names, channel counts per stride) need to match.

### Verifying input/output names

If you bring your own ONNX exports you MUST align the names. Open the
model in **[Netron](https://netron.app)** (also runs as a CLI):
`netron models/retinaface_mnet025_v2.onnx`. The assumed defaults are

| Kind | Default names (per stride 8/16/32) |
| --- | --- |
| detector input  | `input` |
| detector scores | `face_r8_cls`, `face_r16_cls`, `face_r32_cls` |
| detector bbox   | `face_r8_bbox`, `face_r16_bbox`, `face_r32_bbox` |
| detector lmk    | `face_r8_landm`, `face_r16_landm`, `face_r32_landm` |
| recognizer input / output | `input` / `fc1` |

Override via `ONNXOptions.{DetectorBaseNames,DetectorSuffixes,EmbedderOutputName}`
when wiring your own models. The implementation assumes 2 anchors per
cell for the RetinaFace detector — adjust `AnchorsPerCell` (in
`onnx.go`) for variants with different anchor counts.

### What is currently UNVALIDATED

The ONNX embedder implementation is **structurally complete but only
vetted by `go build`/`go vet`**; no real-image smoke test has been run
yet. Known things that may need a touch-up after first real run:

1. **Channel order / mean subtraction** in `preprocessRetinaFace` —
   the buffer defaults assume BGR with `[104, 117, 123]`. Some PyTorch
   re-exports use RGB or `(x-127.5)/128`.
2. **Detector output shapes** are hard-coded for `det_input_size=640`:
   `(2*A, H, W)`, `(4*A, H, W)`, `(10*A, H, W)` per stride. Different
   input sizes must mirror the change in the matched stride sizes.
3. **Recognizer output name** on buffalosc is `embedding`, not `fc1`.
   It's passed through `ONNXOptions.EmbedderOutputName` — set it in
   `main.go` if the auto-default fails.

The deterministic **stub embedder remains the safe choice** for any
end-to-end CI or smoke test (drop the same image into `persona/` and
`archive/`, expect a duplicate in `finded/`).## Roadmap (out of scope for the skeleton)

- Real RetinaFace + ArcFace ONNX embedder (see above).
- Optional photo-source abstraction (local FS today, smb/WebDAV/Yandex
  Disk later).
- Logging via `log/slog` JSON handler for production deployments.

### SQLite embedding cache

A SQLite-backed cache of extracted face embeddings lives in
`internal/cache/`. When `[pipeline].cache_path` is set and the file
exists (or can be created), `main.go` wraps the chosen embedder with
`cache.NewCached`, so:

- The first run over an archive pays the full RetinaFace+ArcFace
  cost and writes one row per file to `cache.sqlite`.
- Subsequent runs over the same archive look up each file by `path`
  and validate the stored `hash` (SHA1 of `path|mtime-nano|size`).
  Re-running over a directory you haven't touched is effectively
  free — only the cosine scoring stage runs.
- Editing, replacing, or mtime-touching a file rebuilds that ONE row
  (`INSERT OR REPLACE`).
- Files with **no faces detected** are cached too — they're the
  dominant cost on real archives (landscapes, pets, scenery).

The cache uses `modernc.org/sqlite` (pure Go, no CGO), WAL journaling,
and `SetMaxOpenConns(1)` to eliminate `SQLITE_BUSY` between the
pipeline's workers. Cache write failures are logged as warnings and
fall back to the freshly extracted embeddings — they never mask a
result.

Schema (verbatim from `TODO.md` §3):

```sql
CREATE TABLE IF NOT EXISTS photo_cache (
    path       TEXT PRIMARY KEY,
    hash       TEXT NOT NULL,
    faces_count INTEGER NOT NULL,
    embeddings BLOB NOT NULL      -- float32 little-endian, EmbedDim*4 bytes per face
);
```

Resize the cache by deleting the file; the schema is recreated on
the next open. The current build only supports vectors of length512 (the constant `EmbedDim` in `internal/cache/codec.go`); other
dimensions will require updating that constant plus the codec.

### Auto-downloading ONNX models

On startup `main.go` checks both `[ml].detector_model` and
`[ml].embedder_model` paths. If either is missing **and**
`[ml].auto_download` is enabled (default: true), the binary fetches
them from `[ml].detector_model_url` /
`[ml].embedder_model_url` — by default a HuggingFace mirror of
insightface's `buffalo_s` pack (the upstream small MobileNet +
MobileFaceNet weight set), which serves the
small MobileNet detector and the MobileFaceNet recognizer as raw
ONNX binary.

Mechanics (`internal/assetfetch.Ensure`):

- Streams each file with a progress `slog` line every ~2 s.
- Writes to `dst + ".tmp"`, then atomic-renames on success.
- Rejects responses that don't begin with the ONNX protobuf header
  (first byte `0x08`) or fall below a per-file size floor (1 MB for
  the detector, 5 MB for the recognizer).
- A previous crashed run leaves no `.tmp` behind — `Ensure` removes
  it before starting.

Override behaviour:

- `--no-download` on the CLI, or `auto_download = false` in INI, to
  disable the fetcher for air-gapped runs.
- `--detector-url` / `--embedder-url`, or `detector_model_url` /
  `embedder_model_url` in INI, to point at a custom mirror.

Failure to download is **non-fatal**: `buildEmbedder` detects the
missing file and falls back to the deterministic stub with a warning,
so a broken network never blocks a run — it just degrades it. 



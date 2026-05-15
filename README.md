# TRELLIS.2 — Single-Image to Textured 3D Asset

> Microsoft Research's 4-billion-parameter sparse transformer turning a
> single photo into a complete 3D model with PBR materials, packaged for
> on-device inference with a themed wrapper, persistent gallery, and
> GLB→STL export for 3D printing.

---

## What this demo is

TRELLIS.2 is the second-generation image-to-3D generative model from
Microsoft Research. In one forward pass it converts a single 2D image
into:

- **Geometry** — a sparse voxel field at 512–1,536 resolution (driven by
  the input's complexity) decoded to a textured mesh.
- **PBR materials** — base color, roughness, metallic, and opacity maps
  baked onto the mesh, so the result drops straight into a modern
  rendering pipeline.

The pipeline is **fully local**: the upstream
[`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2) Gradio
app is bundled inside the container, augmented with background removal
([RMBG-2.0 / BiRefNet](https://huggingface.co/briaai/RMBG-2.0)) so users
don't have to mask their inputs by hand. There is **no API call**
anywhere in the request path — every voxel and texel is sampled by the
GPU you're running on.

This demo wraps that core with:

- **A themed cyberpunk frontend** that live-polls model status, GPU
  utilisation, and the latest generation. Gradio is iframed inside it
  and reverse-proxied so its internal config (`/127.0.0.1:7860`) is
  rewritten to public URLs on the fly.
- **A persistent gallery** (up to 20 items) with thumbnails generated
  from the source image. Each entry carries the GLB plus metadata so
  you can revisit, re-export, or delete past generations.
- **GLB → STL conversion** via `trimesh`, surfaced as a one-click
  "EXPORT STL" button for 3D-printing the geometry (textures are
  dropped — STL is geometry-only).
- **A built-in `model-viewer`** for in-page interactive 3D preview of
  any gallery item.

### What to expect when you run it

1. **Cold start (~3–5 min).** The container boots, the wrapper page
   shows an "INITIALIZING" overlay, and Gradio loads the 4B-parameter
   model + RMBG-2.0 weights into VRAM. The status badge flips to
   "MODEL READY" once `/health` reports Gradio is up.
2. **Drag in an image** (or pick one of the bundled examples in the
   Gradio panel). RMBG-2.0 strips the background automatically.
3. **Generate.** A single forward pass produces the voxel field; the
   decoder bakes geometry + PBR maps; the result streams back as a
   textured GLB rendered by `model-viewer`.
4. **Save / export.** The reverse proxy detects every `.glb` served and
   auto-saves it to the gallery. Click **EXPORT STL** to get a
   3D-printable mesh, or **DOWNLOAD GLB** for the full textured asset.

---

## Capabilities (at a glance)

- 4B-parameter sparse transformer (image → voxels → PBR mesh) running
  fully on-device.
- Single-image input; automatic background removal via RMBG-2.0.
- Adaptive voxel resolution **512 → 1,536** per generation.
- PBR material output: base color, roughness, metallic, opacity.
- GLB export with embedded textures (Unity / Unreal / Blender ready).
- One-click GLB → STL conversion for 3D printing (with mesh stats:
  vertex count, faces, watertight check, bounding box in mm).
- Persistent named gallery (up to 20 entries) with auto-generated
  thumbnails and an in-page `model-viewer` previewer.
- Live header readouts: model status and GPU utilisation (the latter
  via an optional ServiceRouter sidecar; no-ops when absent).
- Bundled Caddy reverse-proxy profile for HTTPS termination.

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). It will run on standard
x86_64 NVIDIA Linux hosts as well, but the bundled Dockerfile pins
PyTorch 2.9.1 + CUDA 13.0 wheels because that's the only stable
combination on aarch64 with the GB10's `sm_121` compute capability.

GB10 also forced one substantive code patch: neither `xformers` nor
`flash-attn` ship working kernels for `sm_121`, so a startup script
(`patch_sdpa.py`) injects a **PyTorch native SDPA backend** into
TRELLIS.2's sparse attention module. That backend is portable — it
works on every CUDA architecture — so the same image runs on H100
(`9.0`), L40S / RTX 4090 (`8.9`), RTX 30xx (`8.6`), etc., as long as
you set `TORCH_CUDA_ARCH_LIST` correctly at build time.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support — won't work. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, **≥ 12 GB VRAM** | 16 GB+ recommended for higher voxel resolutions; OOM on 8 GB cards is likely. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~30 GB image + ~10 GB weights | The image is large because five CUDA extensions are compiled from source; HF weights download on first run. |
| RAM / unified memory | ≥ 32 GB recommended | Higher voxel resolutions push host memory too. |
| HuggingFace token | **Required** | Needed to download TRELLIS.2 + RMBG-2.0 weights. Generate at <https://huggingface.co/settings/tokens> with read scope; accept the model licenses on the HF web UI before first launch. |
| API key | None | Generation runs entirely locally. |

**First build is long (~30–60 minutes)** because the Dockerfile compiles
five CUDA extensions from source: `nvdiffrast`, `nvdiffrec`, `CuMesh`,
`FlexGEMM`, and `o-voxel`. Subsequent rebuilds are incremental — the
extensions sit in their own Docker layers and only recompile if you
change `TORCH_CUDA_ARCH_LIST` or upstream dependencies.

---

## Installation (step-by-step)

These instructions assume a fresh Linux box. If you already have Docker
+ the NVIDIA Container Toolkit working, skip to step 4.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this command fails, **fix your NVIDIA driver before going
further** — the rest will not work.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` table you saw on the host. If
this fails, fix it before continuing.

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/trellis2.git
cd trellis2
```

### 5. Create the environment file

```bash
cp .env.example .env
$EDITOR .env                       # at minimum, set HF_TOKEN
```

The most important edit is **`HF_TOKEN`** — without it the first run
fails to download model weights. If you're not on a GB10, also set
`TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (see
Configuration below).

### 6. Build and start

```bash
docker compose up -d --build
```

The first build takes **30–60 minutes** (PyTorch + CUDA 13 base ~3 GB
plus five CUDA extensions compiled from source). Subsequent starts
take ~10 seconds.

### 7. Verify it's healthy

```bash
docker compose ps
# `demo-trellis2` should be running

curl -s http://localhost:8080/health | python3 -m json.tool
```

Expected output once Gradio finishes loading the model (~3–5 min after
container start):

```json
{
  "status": "ok",
  "service": "trellis2",
  "gradio_ready": true
}
```

While `gradio_ready` is `false`, the wrapper page still serves and
shows a "LOADING MODEL" overlay; it flips to "MODEL READY" once
Gradio's `/` returns 200.

### 8. Open the UI

<http://localhost:8080/>

The wrapper page loads first (cyberpunk header + gallery strip + the
Gradio panel iframed in). Drag an image onto the Gradio panel,
generate, and the resulting GLB will render in the viewer and
auto-save into the gallery strip at the top.

### 9. (Optional) Tail the logs

```bash
docker compose logs -f trellis2
```

You should see:

```
[ENTRYPOINT] Starting Gradio app on port 7860...
[ENTRYPOINT] Starting wrapper server on port 8080...
[TRELLIS2-WRAPPER] ... INFO Started server process
... (model loading lines from TRELLIS.2 / Gradio) ...
```

---

## Configuration

All variables can be set in `.env` or exported in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `HF_TOKEN` | _(empty)_ | **Required.** HuggingFace token used to download TRELLIS.2 + RMBG-2.0 weights on first run. |
| `APP_PORT` | `8080` | Browser-facing port for the wrapper UI and API. (Gradio's internal `7860` is never published.) |
| `TORCH_CUDA_ARCH_LIST` | `12.1` | Compute capability for CUDA-extension compilation. Triggers a full rebuild of the five CUDA extensions when changed. |
| `GRADIO_ROOT_PATH` | `/trellis2/gradio` | URL prefix Gradio is told it lives behind. Only change when proxying under a different mount point. |
| `SERVICEROUTER_URL` | `http://demo-servicerouter:8080` | Optional GPU-stats source for the header readout. The wrapper no-ops cleanly when unreachable. |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port (proxy profile). |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port (proxy profile). |

### Build-time arguments

If you're not on a GB10 / `sm_121` GPU, set `TORCH_CUDA_ARCH_LIST` in
`.env` **before building** so the CUDA extensions compile for your
hardware:

```bash
# In .env:
TORCH_CUDA_ARCH_LIST=8.9    # RTX 40xx / L40S
docker compose build
docker compose up -d
```

Common values: `8.0` (A100), `8.6` (RTX 30xx), `8.9` (RTX 40xx /
L40S), `9.0` (H100), `12.1` (GB10 Grace Blackwell).

### Persistent volumes

| Volume | Purpose |
|---|---|
| `trellis2-hf-cache` | HuggingFace model weights (~10 GB). Survives `docker compose down`. |
| `trellis2-triton-cache` | Triton kernel cache. |
| `trellis2-torch-cache` | PyTorch hub cache. |
| `trellis2-gallery` | Saved GLBs + thumbnails + `gallery.json` index. |
| `trellis2-caddy-data` / `trellis2-caddy-config` | Caddy ACME / runtime state (proxy profile only). |

`docker compose down` keeps these volumes; `docker compose down -v`
removes them (you'll lose your gallery and re-download model weights).

---

## Live controls (in the browser)

The header bar exposes:

- **Status indicator** — INITIALIZING (red) → LOADING MODEL (amber) →
  MODEL READY (green) as Gradio comes up.
- **GPU utilisation bar** — live percentage from ServiceRouter when
  available; static at 0% otherwise.
- **EXPORT STL** — appears once a GLB has been generated. Calls
  `/api/convert-stl`, returns a downloadable STL plus mesh stats
  (vertices, faces, watertight check, bounding box in mm).

The collapsible **gallery strip** (top of page):

- Auto-saves every generated GLB up to a cap of **20 items** (oldest
  evicted first).
- Click any thumbnail to open the in-page `model-viewer` overlay with
  camera controls and auto-rotate.
- From the viewer overlay you can re-export STL or download the GLB.
- Per-card delete button removes the item and its files.

The **Gradio panel** (the iframed core) provides the underlying
TRELLIS.2 controls — image upload, voxel resolution slider, seed,
generate button. All of that ships from upstream.

---

## External services (BYO)

This demo has **no shared dependencies** — the image is fully
self-contained (no Redis, no LiteLLM, no MCP bridge). The BYO override
file exists for compose-flag consistency with the other demos:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` is intentionally an empty `services: {}`.

If you happen to be running ServiceRouter elsewhere on your network and
want the GPU bar to populate, point at it:

| Variable | Example |
|---|---|
| `SERVICEROUTER_URL` | `http://servicerouter.example.com:8080` |

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=trellis.example.com docker compose --profile proxy up -d
```

For local testing keep `DEMO_HOSTNAME=localhost` and Caddy will issue
a self-signed cert.

---

## Authentication

TRELLIS.2 runs **without authentication** by default. Generation is
**GPU-expensive** (each request occupies the GPU for tens of seconds);
do **not** expose this to the public internet without an auth layer.
Options that work in front of the wrapper port:

- **Caddy basic auth** — add a `basic_auth` block to the bundled
  Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

Note: the underlying Gradio server is bound to `0.0.0.0:7860` *inside*
the container but the port is **not published** to the host — only the
wrapper on `${APP_PORT}` is reachable from outside. All Gradio traffic
flows through the wrapper's reverse proxy, which lets the wrapper
intercept GLB downloads for gallery auto-save and rewrite Gradio's
internal URLs to public ones.

---

## Architecture (file map)

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 13.0 devel base, PyTorch 2.9.1 + cu130, five CUDA extensions compiled from source, transformers <5 pin (RMBG-2.0 incompatibility), wrapper assets. |
| `entrypoint.sh` | Runtime patch for RMBG-2.0's `birefnet.py` (forces `device='cpu'` on `torch.linspace` to avoid meta-tensor crashes), then starts Gradio on `7860` and the FastAPI wrapper on `8080`. |
| `patch_sdpa.py` | **Build-time** patch applied to upstream TRELLIS.2: (1) adds `'sdpa'` as a valid sparse-attention backend, (2) implements the SDPA dispatch branch using `torch.nn.functional.scaled_dot_product_attention` for variable-length sparse sequences, (3) sets `low_cpu_mem_usage=False` on the BiRefNet loader. This is what makes TRELLIS.2 run on GB10 / `sm_121` (no `xformers` / `flash-attn` kernels for that arch). |
| `server.py` | FastAPI wrapper. Owns `/health`, `/api/status`, `/api/system/stats` (proxies ServiceRouter), `/api/convert-stl`, `/api/download-stl`, `/api/latest-glb`, the gallery CRUD (`/api/gallery[...]`), and the HTTP + WebSocket reverse proxy to Gradio. The proxy rewrites Gradio's embedded `127.0.0.1:7860` root URL to the public hostname and intercepts `.glb` downloads to drive gallery auto-save. |
| `wrapper/index.html` | Themed cyberpunk shell with header, gallery strip, Gradio iframe, STL panel, and gallery `model-viewer` overlay. |
| `wrapper/static/js/trellis2.js` | Polls `/api/status`, `/api/system/stats`, and `/api/latest-glb`; manages the gallery, STL conversion modal, and the in-page 3D viewer. |
| `wrapper/static/css/trellis2.css` | Cyberpunk theme for the wrapper shell. |
| `wrapper/static/css/gradio-theme.css` | CSS injected into Gradio's `<head>` by the proxy so the iframed UI matches the wrapper. |
| `docker-compose.yml` | Single `trellis2` service + optional `caddy` proxy profile + named volumes. |
| `docker-compose.byo.yml` | Empty `services: {}` — kept for flag-compatibility with other demos. |
| `Caddyfile` | One-liner reverse proxy to `trellis2:8080`. |
| `.env.example` | Template for required + optional env vars. |

The TRELLIS.2 source itself is cloned at build time into
`/app/TRELLIS.2` (not vendored in this repo), patched by
`patch_sdpa.py`, and served by its own `app.py` on port `7860` inside
the container.

---

## Troubleshooting

- **`nvidia-smi` works on host but not in container** — the NVIDIA
  Container Toolkit isn't wired into Docker. Run `sudo nvidia-ctk
  runtime configure --runtime=docker && sudo systemctl restart docker`
  and re-test step 3.
- **First build dies in a CUDA extension** with `nvcc fatal :
  Unsupported gpu architecture 'compute_121'` (or similar) — your
  `TORCH_CUDA_ARCH_LIST` doesn't match your GPU. Fix `.env` and
  rebuild. Common values: `8.6` (RTX 30xx), `8.9` (RTX 40xx / L40S),
  `9.0` (H100), `12.1` (GB10).
- **`HFValidationError` / `401` from HuggingFace** on first start —
  `HF_TOKEN` is unset, expired, or you haven't accepted the upstream
  model licenses. Visit the TRELLIS.2 and RMBG-2.0 model pages on
  huggingface.co, accept the terms, then regenerate a read-scoped
  token.
- **BiRefNet crash with `Cannot copy out of meta tensor`** — the
  `entrypoint.sh` patches the RMBG-2.0 source at container start, but
  the patch only applies if the file already exists in the HF cache.
  On a truly fresh container, the file is downloaded *after*
  entrypoint runs; in that case the build-time patch in `patch_sdpa.py`
  (which sets `low_cpu_mem_usage=False`) is what protects you. If you
  hit it anyway, restart the container — the entrypoint patch will
  apply on the second boot.
- **CUDA OOM during generation** — the model needs 12+ GB of VRAM at
  default settings. Drop the requested voxel resolution in the Gradio
  panel; try a smaller image; or move to a larger GPU. Sparse
  attention + PBR rendering at high resolution is the ceiling.
- **Build cache busted on every build** — keep `TORCH_CUDA_ARCH_LIST`
  stable in `.env`; changing it forces all five CUDA extension layers
  to recompile (~30–60 min).
- **`HF_TOKEN` correct but downloads still fail** — check disk space
  in the `trellis2-hf-cache` volume; weights are ~10 GB.
- **GLB downloads return 404** — the wrapper detects GLBs by watching
  reverse-proxy traffic. If you bypass the wrapper (e.g. hit Gradio
  directly via some other route), the auto-save tracker doesn't fire
  and `/api/latest-glb` returns null.
- **Wrapper page loads but Gradio panel stays blank** — the model is
  still loading. Watch `docker compose logs -f trellis2`; cold start
  is 3–5 minutes. The header status badge will flip from "LOADING
  MODEL" to "MODEL READY" once Gradio's `/` returns 200.
- **`xformers` / `flash-attn` errors on a non-GB10 host** — make sure
  `ATTN_BACKEND=sdpa` is still set (it's hard-coded in the
  `Dockerfile` and `docker-compose.yml`, so this should only matter if
  you've overridden it). The patched SDPA backend is portable across
  all CUDA archs.
- **STL export reports `Invalid or missing GLB file path`** — the
  conversion endpoint allow-lists three roots
  (`/app/TRELLIS.2/tmp`, `/app/gallery`, `/tmp/gradio`); files outside
  these are rejected. This is intentional — don't try to convert
  arbitrary paths.

---

## FAQ

**Q: Can I use a CPU?** No. The 4B sparse transformer plus PBR decoding
is GPU-bound; without CUDA the model won't even load.

**Q: How long does a single generation take?** Tens of seconds on a
modern GPU at default resolution. Higher voxel resolutions and larger
inputs scale roughly linearly.

**Q: Can I batch multiple images?** Not via the bundled UI — the
upstream Gradio app is single-image. The underlying model supports
batching; you'd need to call it directly (see `/app/TRELLIS.2`).

**Q: Are PBR materials preserved in the STL export?** No — STL is a
geometry-only format. Use the GLB if you need textures.

**Q: How big is the gallery?** 20 items max, oldest evicted first.
Stored in the `trellis2-gallery` named volume.

**Q: Can I use my own background-removal model instead of RMBG-2.0?**
Not without code changes. The RMBG-2.0 integration lives in
`/app/TRELLIS.2/trellis2/pipelines/rembg/BiRefNet.py` (patched by
`patch_sdpa.py`).

---

## Credits

Built by Andrew Meinecke.

The underlying image-to-3D model is **TRELLIS.2** by Microsoft Research
(<https://github.com/microsoft/TRELLIS.2>), licensed under its own
terms. Background removal is provided by **RMBG-2.0 / BiRefNet** from
[briaai](https://huggingface.co/briaai/RMBG-2.0). CUDA extensions
([`nvdiffrast`](https://github.com/NVlabs/nvdiffrast),
[`nvdiffrec`](https://github.com/JeffreyXiang/nvdiffrec),
[`CuMesh`](https://github.com/JeffreyXiang/CuMesh),
[`FlexGEMM`](https://github.com/JeffreyXiang/FlexGEMM), and `o-voxel`
from the TRELLIS.2 repo) are compiled from their respective upstreams
at build time and remain under their original licenses.

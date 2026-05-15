"""TRELLIS.2 — Cyberpunk Wrapper Server.

Thin FastAPI gateway that serves the themed frontend, reverse-proxies
all Gradio traffic to the internal Gradio app on port 7860, and provides
a GLB→STL conversion endpoint for 3D printing export.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import urllib.parse
import uuid

import httpx
import trimesh
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.websockets import WebSocketState

SERVICEROUTER_URL = os.environ.get("SERVICEROUTER_URL", "http://demo-servicerouter:8080")
GRADIO_INTERNAL = os.environ.get("GRADIO_INTERNAL_URL", "http://127.0.0.1:7860")
GRADIO_ROOT_PATH = os.environ.get("GRADIO_ROOT_PATH", "/trellis2/gradio")
WRAPPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrapper")
# TRELLIS.2 stores generated files in /app/TRELLIS.2/tmp/
TRELLIS_TMP = "/app/TRELLIS.2/tmp"
GALLERY_DIR = os.environ.get("GALLERY_DIR", "/app/gallery")
GALLERY_MAX_ITEMS = 20

# Server-side GLB tracking — updated by the reverse proxy when a .glb is served.
# Replaces the broken client-side postMessage bridge (Gradio 6.x injects custom
# <head> content too late for fetch interception to work with model-viewer).
_latest_glb = {"path": None, "image_path": None, "timestamp": 0}

# Gradio embeds "root":"http://127.0.0.1:7860/trellis2/gradio" in its config.
# The browser can't reach the internal URL, so we rewrite it to the public URL.
# Gradio JS uses new URL(root) which requires a full URL with scheme/host.
_INTERNAL_ROOT = f"{GRADIO_INTERNAL}{GRADIO_ROOT_PATH}".encode()

# Gradio 6.x JS client strips the root URL to origin-only during _resolve_config(),
# then constructs API URLs as ${origin}${api_prefix}/endpoint.  With the default
# api_prefix="/gradio_api", calls go to https://host/gradio_api/... which doesn't
# match Traefik's PathPrefix(/trellis2).  Rewrite api_prefix to include the full
# path so API calls route correctly through Traefik → FastAPI proxy → Gradio.
_INTERNAL_API_PREFIX = b'"api_prefix":"/gradio_api"'
_PUBLIC_API_PREFIX = f'"api_prefix":"{GRADIO_ROOT_PATH}/gradio_api"'.encode()

# Inject cyberpunk theme CSS into Gradio's <head> (served from the wrapper's
# /static/ mount, accessible in the iframe via the /trellis2/ Traefik route).
_THEME_LINK = b'<link rel="stylesheet" href="/trellis2/static/css/gradio-theme.css">\n</head>'


def _rewrite_gradio_body(body: bytes, request: Request) -> bytes:
    """Replace internal Gradio root URL and api_prefix with public-facing values."""
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    public_root = f"{proto}://{host}{GRADIO_ROOT_PATH}".encode()
    body = body.replace(_INTERNAL_ROOT, public_root)
    body = body.replace(_INTERNAL_API_PREFIX, _PUBLIC_API_PREFIX)
    body = body.replace(b"</head>", _THEME_LINK)
    return body

# Headers to strip from proxied responses (hop-by-hop + encoding that
# conflicts with Starlette's own framing)
STRIP_RESPONSE_HEADERS = frozenset({
    "transfer-encoding", "connection", "content-encoding", "content-length",
})

logging.basicConfig(
    level=logging.INFO,
    format="[TRELLIS2-WRAPPER] %(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("trellis2-wrapper")

app = FastAPI(title="TRELLIS.2", docs_url=None, redoc_url=None)


# ── Health & Status ────────────────────────────────────────────

@app.get("/health")
async def health():
    gradio_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{GRADIO_INTERNAL}/")
            gradio_ok = r.status_code == 200
    except Exception:
        pass
    return {"status": "ok", "service": "trellis2", "gradio_ready": gradio_ok}


@app.get("/api/status")
async def api_status():
    gradio_ready = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{GRADIO_INTERNAL}/")
            gradio_ready = r.status_code == 200
    except Exception:
        pass
    return {"gradio_ready": gradio_ready}


@app.get("/api/system/stats")
async def api_system_stats():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{SERVICEROUTER_URL}/api/system/stats")
            return r.json()
    except Exception as e:
        return {"gpu_percent": None, "error": str(e)}


# ── GLB→STL Conversion ────────────────────────────────────────

def _validate_path(path: str, extension: str) -> str | None:
    """Validate a file path is within allowed directories and has expected extension."""
    if not path:
        return None
    real_path = os.path.realpath(path)
    allowed_roots = [
        os.path.realpath(TRELLIS_TMP),
        os.path.realpath(GALLERY_DIR),
        os.path.realpath("/tmp/gradio"),
    ]
    if not any(real_path.startswith(root) for root in allowed_roots):
        return None
    if not real_path.lower().endswith(extension):
        return None
    if not os.path.exists(real_path):
        return None
    return real_path


@app.post("/api/convert-stl")
async def convert_stl(request: Request):
    """Convert a GLB file to STL for 3D printing."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    glb_path = body.get("glb_path", "")
    gallery_id = body.get("gallery_id", "")
    if gallery_id:
        glb_path = os.path.join(GALLERY_DIR, gallery_id, "model.glb")
    validated = _validate_path(glb_path, ".glb")
    if not validated:
        return JSONResponse(
            {"error": "Invalid or missing GLB file path"},
            status_code=400,
        )

    try:
        scene = trimesh.load(validated)

        if isinstance(scene, trimesh.Scene):
            if len(scene.geometry) == 0:
                return JSONResponse({"error": "GLB contains no geometry"}, status_code=422)
            mesh = trimesh.util.concatenate(list(scene.geometry.values()))
        else:
            mesh = scene

        vertices = int(len(mesh.vertices))
        faces = int(len(mesh.faces))
        is_watertight = bool(mesh.is_watertight)
        bounding_box = mesh.bounding_box.extents.tolist()

        stl_path = validated.rsplit(".glb", 1)[0] + ".stl"
        mesh.export(stl_path, file_type="stl")

        glb_size = os.path.getsize(validated)
        stl_size = os.path.getsize(stl_path)

        logger.info(
            "STL conversion: %d verts, %d faces, %.1f MB GLB → %.1f MB STL",
            vertices, faces, glb_size / 1e6, stl_size / 1e6,
        )

        return {
            "download_url": "/trellis2/api/download-stl?path=" + urllib.parse.quote(stl_path),
            "filename": os.path.basename(stl_path),
            "vertices": vertices,
            "faces": faces,
            "is_watertight": is_watertight,
            "bounding_box_mm": bounding_box,
            "glb_size_mb": round(glb_size / (1024 * 1024), 1),
            "stl_size_mb": round(stl_size / (1024 * 1024), 1),
        }
    except Exception as e:
        logger.exception("STL conversion failed")
        return JSONResponse({"error": f"Conversion failed: {e}"}, status_code=500)


@app.get("/api/download-stl")
async def download_stl(path: str):
    """Download a previously converted STL file."""
    validated = _validate_path(path, ".stl")
    if not validated:
        return JSONResponse({"error": "Invalid or missing STL file path"}, status_code=400)
    return FileResponse(
        validated,
        media_type="model/stl",
        filename=os.path.basename(validated),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(validated)}"'},
    )


# ── Server-Side GLB Tracking ──────────────────────────────────

@app.get("/api/latest-glb")
async def get_latest_glb():
    """Return the most recently served GLB path (detected by reverse proxy)."""
    return _latest_glb


# ── Gallery API ──────────────────────────────────────────────

def _gallery_index_path():
    return os.path.join(GALLERY_DIR, "gallery.json")


def _load_gallery() -> list:
    path = _gallery_index_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_gallery(items: list):
    os.makedirs(GALLERY_DIR, exist_ok=True)
    with open(_gallery_index_path(), "w") as f:
        json.dump(items, f, indent=2)


def _generate_thumbnail(image_path: str, output_path: str):
    """Center-crop image to 4:3 and resize to 400x300."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        target_ratio = 4 / 3
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        elif current_ratio < target_ratio:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        img = img.resize((400, 300), Image.LANCZOS)
        img.convert("RGB").save(output_path, "JPEG", quality=85)
        return True
    except Exception as e:
        logger.warning("Thumbnail generation failed: %s", e)
        return False


@app.get("/api/gallery")
async def list_gallery():
    """List all gallery items, newest first."""
    items = _load_gallery()
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"items": items}


@app.post("/api/gallery")
async def save_to_gallery(request: Request):
    """Save a GLB + source image to the persistent gallery."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    glb_path = body.get("glb_path", "")
    image_path = body.get("image_path", "")

    if not glb_path or not os.path.exists(glb_path):
        return JSONResponse({"error": "GLB file not found"}, status_code=400)

    items = _load_gallery()

    # Duplicate prevention by GLB filename
    glb_name = os.path.basename(glb_path)
    if any(item.get("source_filename") == glb_name for item in items):
        existing = next(i for i in items if i.get("source_filename") == glb_name)
        return {"id": existing["id"], "duplicate": True}

    # Enforce max items (remove oldest)
    while len(items) >= GALLERY_MAX_ITEMS:
        oldest = min(items, key=lambda x: x.get("created_at", 0))
        item_dir = os.path.join(GALLERY_DIR, oldest["id"])
        if os.path.isdir(item_dir):
            shutil.rmtree(item_dir, ignore_errors=True)
        items.remove(oldest)

    item_id = uuid.uuid4().hex[:8]
    item_dir = os.path.join(GALLERY_DIR, item_id)
    os.makedirs(item_dir, exist_ok=True)

    # Copy GLB
    dest_glb = os.path.join(item_dir, "model.glb")
    shutil.copy2(glb_path, dest_glb)
    glb_size = os.path.getsize(dest_glb)

    # Generate thumbnail from source image
    has_thumbnail = False
    if image_path and os.path.exists(image_path):
        thumb_path = os.path.join(item_dir, "thumbnail.jpg")
        has_thumbnail = _generate_thumbnail(image_path, thumb_path)

    item = {
        "id": item_id,
        "name": glb_name.rsplit(".", 1)[0],
        "created_at": time.time(),
        "glb_size_mb": round(glb_size / (1024 * 1024), 1),
        "has_thumbnail": has_thumbnail,
        "source_filename": glb_name,
    }
    items.append(item)
    _save_gallery(items)

    logger.info("Gallery: saved %s (%.1f MB)", item_id, item["glb_size_mb"])
    return {"id": item_id, "item": item}


@app.delete("/api/gallery/{item_id}")
async def delete_gallery_item(item_id: str):
    """Remove a gallery item and its files."""
    items = _load_gallery()
    item = next((i for i in items if i["id"] == item_id), None)
    if not item:
        return JSONResponse({"error": "Item not found"}, status_code=404)

    item_dir = os.path.join(GALLERY_DIR, item_id)
    if os.path.isdir(item_dir):
        shutil.rmtree(item_dir, ignore_errors=True)

    items = [i for i in items if i["id"] != item_id]
    _save_gallery(items)
    logger.info("Gallery: deleted %s", item_id)
    return {"deleted": item_id}


@app.get("/api/gallery/{item_id}/model.glb")
async def serve_gallery_glb(item_id: str):
    """Serve a gallery item's GLB file."""
    glb_path = os.path.join(GALLERY_DIR, item_id, "model.glb")
    validated = _validate_path(glb_path, ".glb")
    if not validated:
        return JSONResponse({"error": "GLB not found"}, status_code=404)
    return FileResponse(validated, media_type="model/gltf-binary")


@app.get("/api/gallery/{item_id}/thumbnail.jpg")
async def serve_gallery_thumbnail(item_id: str):
    """Serve a gallery item's thumbnail."""
    thumb_path = os.path.join(GALLERY_DIR, item_id, "thumbnail.jpg")
    if not os.path.exists(thumb_path):
        return JSONResponse({"error": "Thumbnail not found"}, status_code=404)
    return FileResponse(thumb_path, media_type="image/jpeg")


# ── Wrapper Page ───────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(WRAPPER_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(WRAPPER_DIR, "static")), name="static")


# ── HTTP Reverse Proxy (Gradio) ───────────────────────────────
# IMPORTANT: HTTP routes MUST be registered BEFORE WebSocket routes.
# Starlette matches routes in registration order; a WebSocket route
# will return 405 for HTTP requests if it matches first.

@app.get("/gradio")
async def proxy_gradio_root(request: Request):
    """Proxy the bare /gradio path."""
    url = f"{GRADIO_INTERNAL}/"
    if request.query_params:
        url += f"?{request.query_params}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        response_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in STRIP_RESPONSE_HEADERS
        }
        body = _rewrite_gradio_body(resp.content, request)
        return StreamingResponse(
            iter([body]),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )


@app.api_route(
    "/gradio/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_gradio(request: Request, path: str):
    """Reverse proxy all HTTP requests to Gradio."""
    # Server-side GLB/image detection — track files as they pass through proxy
    if "file=" in path:
        file_part = path.split("file=", 1)[1]
        decoded = urllib.parse.unquote(file_part)
        if decoded.lower().endswith(".glb"):
            _latest_glb["path"] = decoded
            _latest_glb["timestamp"] = time.time()
            logger.info("GLB detected: %s", decoded)
        elif any(decoded.lower().endswith(ext) for ext in (".webp", ".png", ".jpg", ".jpeg")):
            _latest_glb["image_path"] = decoded

    url = f"{GRADIO_INTERNAL}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "connection", "transfer-encoding")
    }

    body = await request.body()

    # For root page (path=""), read full body and rewrite internal URLs
    # so the browser doesn't try to connect to 127.0.0.1:7860.
    if not path:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            req = client.build_request(
                method=request.method, url=url, headers=headers, content=body,
            )
            resp = await client.send(req)
            response_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in STRIP_RESPONSE_HEADERS
            }
            content = _rewrite_gradio_body(resp.content, request)
            return StreamingResponse(
                iter([content]),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

    # For all other paths, stream the response.  The httpx client and
    # response must stay open until Starlette finishes sending, so we
    # manage their lifecycle via an async generator instead of relying
    # on `async with` (which would close the client on return).
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    req = client.build_request(
        method=request.method, url=url, headers=headers, content=body,
    )
    resp = await client.send(req, stream=True)

    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in STRIP_RESPONSE_HEADERS
    }

    async def _stream_and_cleanup():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream_and_cleanup(),
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )


# ── WebSocket Proxy (Gradio queue) ────────────────────────────
# Registered AFTER HTTP routes so it doesn't intercept HTTP requests.

@app.websocket("/gradio/{path:path}")
async def ws_proxy(websocket: WebSocket, path: str):
    """Bidirectional WebSocket proxy to Gradio."""
    await websocket.accept()

    gradio_ws_url = f"ws://127.0.0.1:7860/{path}"
    if websocket.query_params:
        gradio_ws_url += f"?{websocket.query_params}"

    try:
        import websockets
        async with websockets.connect(gradio_ws_url) as gradio_ws:
            async def client_to_gradio():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await gradio_ws.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            async def gradio_to_client():
                try:
                    async for msg in gradio_ws:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_text(msg)
                except Exception:
                    pass

            await asyncio.gather(client_to_gradio(), gradio_to_client())
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass

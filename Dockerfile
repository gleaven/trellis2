ARG CUDA_VERSION=13.0.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_ARCH=12.1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
ENV PYTHONUNBUFFERED=1
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    cmake ninja-build git wget curl build-essential \
    libgl1-mesa-glx libegl1-mesa libglib2.0-0 \
    libx11-dev libglx-dev libegl-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.9.1 stable + cu130 (proven on GB10 sm_121)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.9.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Clone TRELLIS.2 with Eigen submodule
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/TRELLIS.2

# Basic Python dependencies (transformers excluded — installed after CUDA extensions)
RUN pip3 install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh gradio==6.0.1 tensorboard pandas lpips \
    zstandard kornia timm pillow \
    httpx websockets uvicorn fastapi && \
    pip3 install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# ── CUDA Extensions (each ~5-15 min compile) ──

# nvdiffrast v0.4.0 — differentiable rasterization
# Fix: setup.py lacks name/packages, causing UNKNOWN install with --no-build-isolation
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast && \
    sed -i "s/setup(/setup(name='nvdiffrast', version='0.4.0', packages=['nvdiffrast', 'nvdiffrast.torch'], /" /tmp/ext/nvdiffrast/setup.py && \
    pip3 install --no-cache-dir /tmp/ext/nvdiffrast --no-build-isolation && \
    python3 -c "import nvdiffrast.torch; print('[OK] nvdiffrast installed')"

# nvdiffrec — differentiable PBR rendering
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec && \
    pip3 install --no-cache-dir /tmp/ext/nvdiffrec --no-build-isolation

# Verify nvdiffrast still importable after nvdiffrec install
RUN python3 -c "import nvdiffrast.torch; print('[OK] nvdiffrast still importable after nvdiffrec')"

# CuMesh — CUDA-accelerated mesh utilities
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh --recursive && \
    pip3 install --no-cache-dir /tmp/ext/CuMesh --no-build-isolation

# FlexGEMM — Triton-based sparse convolution
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM --recursive && \
    pip3 install --no-cache-dir /tmp/ext/FlexGEMM --no-build-isolation

# O-Voxel — sparse volumetric representation (uses Eigen submodule)
RUN pip3 install --no-cache-dir /app/TRELLIS.2/o-voxel --no-build-isolation

# Cleanup CUDA extension build sources
RUN rm -rf /tmp/ext

# ── Post-extension deps (changing these won't bust CUDA extension cache) ──
# transformers <5 required: RMBG-2.0 custom code incompatible with transformers 5.x
RUN pip3 install --no-cache-dir "transformers<5"

# ── Patch sparse attention AFTER CUDA extensions (keeps extension layers cacheable) ──
COPY patch_sdpa.py /tmp/patch_sdpa.py
RUN python3 /tmp/patch_sdpa.py /app/TRELLIS.2 && rm /tmp/patch_sdpa.py

# Runtime entrypoint patches HF-downloaded model code if needed
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cyberpunk wrapper: FastAPI server + themed HTML/CSS/JS
COPY server.py /app/trellis2-wrapper/server.py
COPY wrapper/ /app/trellis2-wrapper/wrapper/

WORKDIR /app/TRELLIS.2

ENV ATTN_BACKEND=sdpa
ENV GRADIO_ROOT_PATH=/trellis2/gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]

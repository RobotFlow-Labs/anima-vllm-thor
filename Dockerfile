# anima-vllm:thor-latest — latest vLLM (0.23.x) on NVIDIA Jetson AGX Thor (sm_110a)
# Builds vLLM 0.23 + PyTorch 2.11 + CUDA 13, natively for Blackwell sm_110a, on the NVIDIA base.
# Every RUN below corresponds to a numbered gotcha in the README. Build: ~30-50 min, ~90 GB RAM.
#   DOCKER_BUILDKIT=1 docker build -t anima-vllm:thor-latest -f Dockerfile .
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

# (1) the base bakes PIP_CONSTRAINT=torch==2.10.0 → unset it or torch 2.11 can't install
ENV PIP_CONSTRAINT="" \
    TORCH_CUDA_ARCH_LIST=11.0a \
    FLASHINFER_CUDA_ARCH_LIST=11.0a \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=12 \
    VLLM_TARGET_DEVICE=cuda \
    CMAKE_BUILD_TYPE=Release

# (2) remove vLLM 0.19 (pins torch==2.10) + old flashinfer before upgrading torch
RUN pip uninstall -y vllm flashinfer-python flashinfer-cubin flashinfer-jit-cache || true

# (3) torch 2.11.0+cu130 — official aarch64 wheel, verified to carry sm_110 kernels
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
 && (pip install --no-cache-dir torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 || true)

# (4) build vLLM 0.23 from source against the installed torch 2.11 (reuse it, strip torch pins)
WORKDIR /build
RUN git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git
WORKDIR /build/vllm
RUN python use_existing_torch.py || true
RUN pip install --no-cache-dir "setuptools-scm>=8"
RUN MAX_JOBS=${MAX_JOBS} pip wheel --no-build-isolation --no-deps -w /build/dist .
RUN set -eux; WHL="$(find /build -name 'vllm-*.whl' | head -1)"; echo "WHEEL=$WHL"; test -n "$WHL"; \
    pip install --no-cache-dir --no-deps "$WHL"

# (5) flashinfer — required by vLLM at engine init; 0.6.8 is the newest on the arm64 index (vLLM wants 0.6.12)
RUN pip install --no-cache-dir flashinfer-python==0.6.8 || true

# (7) compressed-tensors must be 0.17.x (NVFP4); the base's is stale (missing compressors.pack_quantized)
RUN pip install --no-cache-dir "compressed-tensors==0.17.1"

# (8) vLLM 0.23 eagerly imports ALL quant backends; guard the ones not packaged for arm64
#     (humming/inc/fp_quant/torchao/quark/deepseek_v4/mxfp4) so NVFP4/compressed-tensors still load
COPY scripts/patch_quant.py /tmp/patch_quant.py
RUN python /tmp/patch_quant.py

# (6) the cloned source tree shadows the installed _C.so → remove it and serve from /
RUN rm -rf /build
WORKDIR /
RUN python -c "import vllm, torch; print('BUILT vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || true

# (5b) RUNTIME-ONLY: preload the real driver libcuda — vLLM 0.23's _C_stable_libtorch.abi3.so has an
#      unresolved cuTensorMapEncodeTiled symbol. The driver lib is injected by `--runtime nvidia` at run
#      time, so this MUST be the final instruction (no RUN after it).
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1

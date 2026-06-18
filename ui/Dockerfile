# anima-thor-ui control plane — runs as a long-lived, auto-restarting container.
# It drives SIBLING containers (the vLLM engine, the quantize worker) via the host
# docker socket, so it ships the docker CLI and is run with the socket mounted.
# App code + HF cache are bind-mounted at IDENTICAL host paths at runtime so the
# nested `docker run -v <host-path>` commands the UI issues resolve correctly.
FROM python:3.12-slim

# Official Docker CLI client (docker-ce-cli) — debian's `docker.io` ships the daemon
# but not the `docker` client binary. This installs a glibc-matched client at /usr/bin/docker.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn[standard]>=0.34" "httpx>=0.27" \
    "huggingface-hub>=0.27" "pydantic>=2.9" "python-multipart>=0.0.12"

# Code is bind-mounted at run time (same path in/out) — nothing baked, so edits
# deploy with a simple rsync + container restart. WORKDIR/CMD set by the launcher.
EXPOSE 7000
CMD ["python", "-m", "anima_thor_ui.main"]

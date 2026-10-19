FROM python:3.13-slim

WORKDIR /app

ARG GITHUB_TOKEN

RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm ca-certificates && rm -rf /var/lib/apt/lists/*

# Pin the app-server protocol version used by the subscription transport.
ARG CODEX_VERSION=0.154.0
RUN npm install --global @openai/codex@${CODEX_VERSION} && npm cache clean --force
ENV RECORDATIO_CODEX_HOME=/root/.codex-memento

COPY pyproject.toml requirements.txt .
COPY src/ src/
COPY scripts/smoke_codex_subscription.py scripts/smoke_codex_mcp.py scripts/
# Install CPU-only torch first (the deploy host has no GPU): the default PyPI torch
# wheel on Linux bundles ~multi-GB of CUDA libs the bi-encoder never uses. The
# +cpu build from PyTorch's CPU index satisfies requirements.txt's torch==2.12.0
# (local-version match) so the subsequent install skips it.
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/" && \
    pip install --no-cache-dir torch==2.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt -e . && \
    git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf

# Prefetch the F2 bi-encoder (BAAI/bge-small-en-v1.5, ~130MB) into the image so
# there is no cold download at runtime. Local-pinned, offline at serve time,
# cached under the default HF hub dir. The #42 cross-encoder reranker is NOT
# prefetched: it has no production caller (fuse_rrf is bench-only), so shipping
# its ~280MB weights would only bloat the image.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8787

CMD ["agentmem"]

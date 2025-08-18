FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git git-lfs libsndfile1 tini \
    && git lfs install --skip-smudge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fast installer
RUN pip install -U uv

# Install deps first (good layer caching)
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

# Copy app code
COPY . .
RUN mkdir -p /app/media /app/output

EXPOSE 8501
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

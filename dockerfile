FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs libsndfile1 tini && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /app

# Avoid pip cache
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Folders your app writes to
RUN mkdir -p /app/media /app/output

# Streamlit defaults
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/.cache/huggingface

# Talk to Ollama on host
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8501
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

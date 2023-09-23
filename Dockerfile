FROM python:3.9-slim-bullseye AS base
WORKDIR /app

FROM base AS build
COPY ["requirements.txt", "./"]
RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    g++ && \
    rm -rf /var/lib/apt/lists/* && \
    # Install dependencies.
    python -m venv ./venv && \
    ./venv/bin/pip install --upgrade \
    pip \
    setuptools \
    wheel && \
    ./venv/bin/pip install --no-cache-dir -r ./requirements.txt && \
    ./venv/bin/pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@d1e04565d3bec8719335b88be9e9b961bf3ec464'

FROM base AS final
RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build ["/app/venv", "./venv"]
# Copy the source code in last to optimize rebuilding the image.
COPY [".", "./"]
ENTRYPOINT \
    # Download important nltk data packages.
    ./venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    ./venv/bin/python download_models.py && \
    ./venv/bin/python -m gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080
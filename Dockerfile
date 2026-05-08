ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple torch

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY tests ./tests

RUN python -m pip install -e ".[dev]" \
    && mkdir -p /app/runs

VOLUME ["/app/runs"]

CMD ["idp-train", "--config", "configs/default.yaml", "--run-dir", "runs/docker_smoke", "--total-steps", "2048", "--rollout-steps", "256", "--device", "cpu"]

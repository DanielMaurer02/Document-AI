#TODO: Optimize image for arm64, currently over 1 GB bigger than amd64


FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

ENV UV_PYTHON_DOWNLOADS=0


ARG EMBEDDING_SERVICE
ARG LLM_SERVICE
ARG EMBEDDING_MODEL_NAME
ARG LLM_MODEL_NAME
ARG CHROMA_HOST
ARG CHROMA_PORT
ARG DOMAIN

# Set environment variables for configuration
ENV EMBEDDING_SERVICE=${EMBEDDING_SERVICE}
ENV LLM_SERVICE=${LLM_SERVICE}
ENV EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}
ENV CHROMA_HOST=${CHROMA_HOST}
ENV CHROMA_PORT=${CHROMA_PORT}
ENV LLM_MODEL_NAME=${LLM_MODEL_NAME}
ENV DOMAIN=${DOMAIN}
ENV TOKENIZERS_PARALLELISM=False

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.13-slim-bookworm
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]

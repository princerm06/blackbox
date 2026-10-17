FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY blackbox ./blackbox

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir ".[open-vocab]"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn blackbox.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

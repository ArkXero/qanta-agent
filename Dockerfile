FROM python:3.12-slim AS index-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
RUN pip install --no-cache-dir pyarrow==24.0.0
COPY scripts ./scripts
RUN python scripts/build_index.py --raw-dir /tmp/qanta_raw --out /build/data/qanta_index.sqlite


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QANTA_INDEX_PATH=/app/data/qanta_index.sqlite \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent ./agent
COPY --from=index-builder /build/data/qanta_index.sqlite ./data/qanta_index.sqlite

EXPOSE 8080
CMD ["uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "8080"]

# QANTA Docker Agent

Docker-ready QANTA 2026 agent implementing:

- `GET /health`
- `POST /predict/tossup`
- `POST /predict/bonus`

The current system is a self-contained SQLite FTS retrieval agent built from public historical QANTA/quizbowl data. It does not call external APIs at prediction time.

## Build Index

```bash
uv run --with pyarrow scripts/build_index.py
```

## Smoke Test

```bash
uv run scripts/smoke_test.py
```

## Run Server

```bash
uv run --with fastapi --with uvicorn uvicorn agent.server:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker build -t qanta-agent:latest .
docker run --rm -p 8080:8080 qanta-agent:latest
```

Submit a public registry image reference in the QANTA Docker Submissions tab after pushing.

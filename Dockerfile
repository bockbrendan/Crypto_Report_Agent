# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy only the application code — never copy .env
COPY intel_agent/ ./intel_agent/
COPY api/         ./api/
COPY backend/     ./backend/
COPY frontend/    ./frontend/

# Run as non-root for security
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# One worker: graph is a stateless singleton; concurrent SSE requests each
# run in their own thread with an isolated state dict — fully thread-safe.
CMD ["uvicorn", "backend.agents.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

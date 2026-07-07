# ============================================================================
# Dockerfile = recipe for building an IMAGE (a frozen snapshot of a machine
# that has exactly what your app needs). A running copy of an image is a
# CONTAINER. Same image runs identically on your laptop, a server, anywhere —
# that's the whole point: "works on my machine" becomes "works everywhere".
#
# Build:  docker build -t lexmacedonica .
# Run:    docker compose up        (see docker-compose.yml)
# ============================================================================

# Start FROM a base image: official Python on slim Debian Linux.
# We use 3.12 (not 3.14 like your Windows venv) — inside the container we
# control the world, so we pick the version with the most mature library
# support. The code runs on both.
FROM python:3.12-slim

# Everything below happens INSIDE the image, in this folder:
WORKDIR /app

# Copy ONLY requirements first, then install. Docker caches each step
# (a "layer") — as long as requirements.txt doesn't change, rebuilds skip
# the slow pip install and only re-copy your changed code. Order matters!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code (.dockerignore keeps data/, .venv etc. out).
COPY . .

# Document that the app listens on port 8000 (informational).
EXPOSE 8000

# What runs when the container starts.
# --host 0.0.0.0 is REQUIRED in containers: it means "accept connections from
# outside the container", while the default 127.0.0.1 would only accept
# connections from inside it (and you could never reach the app).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

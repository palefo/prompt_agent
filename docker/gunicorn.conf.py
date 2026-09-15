"""Gunicorn settings for the Docker image (tunable through environment variables)."""
import os

bind = "0.0.0.0:8000"
# Threads suit this app: most of a request is spent waiting for the LLM.
worker_class = "gthread"
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
# Must stay above LLM_TIMEOUT x (retries + 1).
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "240"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
# Only Caddy can reach the container, so trust its X-Forwarded-* headers.
forwarded_allow_ips = "*"

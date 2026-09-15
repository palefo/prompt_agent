# AI Futures study platform (Flask + LangChain), served by gunicorn.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Unprivileged user that owns uploaded study files. UID 1000 matches the first
# regular user on most Linux hosts, so the bind-mounted data folder stays editable.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
# *.mo files are not committed, so compile the translation catalogues here.
RUN pybabel compile -d translations \
    && chmod 0755 docker/entrypoint.sh

ENV STUDIES_DIR=/data/studies \
    SEED_STUDIES_FROM=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["gunicorn", "--config", "docker/gunicorn.conf.py", "app:app"]

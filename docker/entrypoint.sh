#!/bin/sh
# Prepare the persistent studies folder, copy the bundled studies into it on
# first start, then run the app as the unprivileged "app" user.
set -eu

STUDIES_DIR="${STUDIES_DIR:-/data/studies}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$STUDIES_DIR"
    chown app:app "$STUDIES_DIR"
    exec setpriv --reuid=app --regid=app --init-groups "$0" "$@"
fi

export HOME="${HOME:-/home/app}"
[ -w "$HOME" ] || export HOME=/home/app

python -m backend.studies seed
exec "$@"

#!/bin/sh
# Back up the database and the study files into backups/<timestamp>/.
#
#   scripts/backup.sh               # keeps the newest 14 backups
#   KEEP=30 scripts/backup.sh
#
# Restore (see DEPLOYMENT.md):
#   docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists' < backups/<stamp>/database.dump
#   tar -xzf backups/<stamp>/studies.tar.gz -C data/
set -eu
cd "$(dirname "$0")/.."

KEEP="${KEEP:-14}"
BACKUP_ROOT="${BACKUP_DIR:-./backups}"
DEST="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom' > "$DEST/database.dump"
docker compose exec -T app tar -czf - -C /data studies > "$DEST/studies.tar.gz"

# Sanity check: both files must be non-empty.
[ -s "$DEST/database.dump" ] && [ -s "$DEST/studies.tar.gz" ] || { echo "Backup failed: empty file in $DEST" >&2; exit 1; }

echo "$(date -Is) backup written to $DEST ($(du -sh "$DEST" | cut -f1))"

# Keep only the newest $KEEP backups.
ls -1dt "$BACKUP_ROOT"/*/ 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -rf

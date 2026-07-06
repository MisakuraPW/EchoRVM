#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/root/autodl-fs/augmented}"
DST="${2:-/root/autodl-tmp/augmented}"

mkdir -p "${DST}"
rsync -ah --info=progress2 "${SRC}/" "${DST}/"

echo "Synced augmented data:"
echo "  from ${SRC}"
echo "  to   ${DST}"

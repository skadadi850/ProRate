#!/usr/bin/env bash
# Downloads the checkpoints, test sets and cached baseline results from Zenodo into release/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ZENODO_RECORD="REPLACE_WITH_RECORD_ID"
SHA256="7ccaa3c5c7b70263ecd68c8add846c656498dde9e1599de1f9fa3d5ce0b9fdf3"
FILE=prorate_release.tar.gz
curl -L --fail -o "$FILE" "https://zenodo.org/records/$ZENODO_RECORD/files/$FILE?download=1"
echo "$SHA256  $FILE" | sha256sum -c -
tar -xzf "$FILE"
rm "$FILE"
echo "release/ ready"
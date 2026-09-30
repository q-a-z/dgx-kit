#!/usr/bin/env bash
# Builds the release package from the last commit: dist/dgx-kit-<version>.tar.gz and its checksum.
set -euo pipefail
cd "$(dirname "$0")/.."
git diff --quiet HEAD -- || echo "note: uncommitted changes are not in the package (it is built from the last commit)" >&2
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
NAME=dgx-kit-$VERSION
mkdir -p dist
git archive --format=tar.gz --prefix="$NAME/" -o "dist/$NAME.tar.gz" HEAD
( cd dist && { sha256sum "$NAME.tar.gz" 2>/dev/null || shasum -a 256 "$NAME.tar.gz"; } > "$NAME.tar.gz.sha256" )
echo "dist/$NAME.tar.gz"
cat "dist/$NAME.tar.gz.sha256"

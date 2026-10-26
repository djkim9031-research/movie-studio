#!/usr/bin/env bash
# Installs the pinned gitleaks release after checking its SHA-256. Releases
# are not signed, so the hash pinned here is what the install trusts.
# Usage: scripts/hygiene/install-gitleaks.sh [dest-dir], default ~/.local/bin
set -euo pipefail
version=8.30.1
sha256=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
asset=gitleaks_${version}_linux_x64.tar.gz
dest=${1:-$HOME/.local/bin}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/$asset" "https://github.com/gitleaks/gitleaks/releases/download/v$version/$asset"
echo "$sha256  $tmp/$asset" | sha256sum --check --quiet
tar -xzf "$tmp/$asset" -C "$tmp" gitleaks
mkdir -p "$dest"
install -m 0755 "$tmp/gitleaks" "$dest/gitleaks"
"$dest/gitleaks" version

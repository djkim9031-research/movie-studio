#!/usr/bin/env bash
# Reads back what is on the remote and checks it: only main and tags may
# exist, and every commit must pass the same checks as pre-push.
# Usage: scripts/hygiene/audit.sh [remote], default origin
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
. .githooks/lib.sh
load_deny
want_identity
need_gitleaks

remote=${1:-origin}
git fetch -q --prune "$remote"
refs=$(git ls-remote "$remote" | awk '{print $2}')
extra=$(printf '%s\n' "$refs" | grep -Ev '^(HEAD|refs/heads/main|refs/tags/.+)$' || true)
[ -z "$extra" ] || fail "unexpected refs on $remote: $(printf '%s\n' "$extra" | paste -sd' ')"
check_range "refs/remotes/$remote/main"
note "audit passed: $(git rev-list --count "refs/remotes/$remote/main") commits on $remote/main; refs: $(printf '%s\n' "$refs" | paste -sd' ')"

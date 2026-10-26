#!/usr/bin/env bash
# Configures this clone: identity, hooks, commit template, credentials and
# local excludes. Everything goes into .git/config or .git/info, nothing global.
# Usage: scripts/hygiene/setup-repo.sh --name NAME --email EMAIL [--gh-user LOGIN]
set -euo pipefail
name='' email='' gh_user=''
while [ $# -gt 0 ]; do
    case $1 in
        --name) name=$2; shift 2 ;;
        --email) email=$2; shift 2 ;;
        --gh-user) gh_user=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$name" ] && [ -n "$email" ] || { echo "usage: $0 --name NAME --email EMAIL [--gh-user LOGIN]" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"

git config --local user.name "$name"
git config --local user.email "$email"
git config --local hygiene.name "$name"
git config --local hygiene.email "$email"
git config --local core.hooksPath .githooks
git config --local commit.template .gitmessage

# GitHub over HTTPS gets this account's token from gh and nothing else: the
# empty value drops every helper set in the global and system config.
if [ -n "$gh_user" ]; then
    gh=$(command -v gh) || { echo "gh not found" >&2; exit 1; }
    key=credential.https://github.com.helper
    git config --local --unset-all "$key" || true
    git config --local --add "$key" ''
    git config --local --add "$key" \
        "!f() { test \"\$1\" = get || exit 0; echo username=$gh_user; echo \"password=\$($gh auth token --user $gh_user)\"; }; f"
fi

excludes=${HYGIENE_EXCLUDES_FILE:-$HOME/studio-data/secrets/repo-local-excludes.txt}
if [ -r "$excludes" ]; then
    info=$(git rev-parse --git-common-dir)/info
    mkdir -p "$info"
    cp "$excludes" "$info/exclude"
fi

command -v gitleaks >/dev/null || echo "gitleaks not found; run scripts/hygiene/install-gitleaks.sh" >&2
git config --local --get-regexp '^(user|hygiene)\.|^core\.hookspath|^commit\.template|^credential\.'

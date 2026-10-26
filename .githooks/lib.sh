# Shared by the hooks and scripts/hygiene. Sourced, not run.

HYGIENE_DENY_FILE=${HYGIENE_DENY_FILE:-$HOME/studio-data/secrets/repo-deny-patterns.txt}
HYGIENE_MAX_BYTES=${HYGIENE_MAX_BYTES:-5242880}

note() { printf '[hygiene] %s\n' "$*" >&2; }
fail() { note "$*"; exit 1; }

# Deny list: one extended regex per line, matched case-insensitively.
# Blank lines and lines starting with # are skipped. Lines starting with
# "sample: " are strings for the self-test, not patterns.
load_deny() {
    [ -f "$HYGIENE_DENY_FILE" ] && [ -r "$HYGIENE_DENY_FILE" ] || fail "deny list missing: $HYGIENE_DENY_FILE"
    DENY=$(grep -Ev '^[[:space:]]*(#|$)|^sample: ' "$HYGIENE_DENY_FILE" || true)
    [ -n "$DENY" ] || fail "deny list has no patterns: $HYGIENE_DENY_FILE"
}

# Reads text on stdin. $1 names it in the message.
check_deny() {
    local hits rc=0
    hits=$(grep -Ein -f <(printf '%s\n' "$DENY")) || rc=$?
    [ "$rc" -le 1 ] || fail "deny list has an invalid pattern"
    if [ -n "$hits" ]; then
        note "deny-list match in $1:"
        printf '%s\n' "$hits" | head -n 5 | cut -c1-200 >&2
        exit 1
    fi
}

want_identity() {
    WANT_NAME=$(git config hygiene.name || true)
    WANT_EMAIL=$(git config hygiene.email || true)
    [ -n "$WANT_NAME" ] && [ -n "$WANT_EMAIL" ] \
        || fail "hygiene.name and hygiene.email are unset; run scripts/hygiene/setup-repo.sh"
}

need_gitleaks() {
    command -v gitleaks >/dev/null || fail "gitleaks not found; run scripts/hygiene/install-gitleaks.sh"
}

# Unified diff on stdin, added lines without the leading + on stdout.
added_lines() {
    { grep -Ev '^\+\+\+ (a/|b/|/dev/null)' || true; } | sed -n 's/^+//p'
}

# Paths on stdin, one per line. Refuses ignored paths and deny-listed names.
check_paths() {
    local paths ignored rc=0
    paths=$(cat)
    [ -n "$paths" ] || return 0
    ignored=$(printf '%s\n' "$paths" | git check-ignore --no-index --stdin) || rc=$?
    [ "$rc" -le 1 ] || fail "git check-ignore failed"
    [ -z "$ignored" ] || fail "ignored path: $(printf '%s\n' "$ignored" | head -n 3 | paste -sd' ')"
    printf '%s\n' "$paths" | check_deny "file names"
}

# Checks every commit selected by the rev-list arguments: authorship,
# merges, messages, file names, file sizes, added lines and gitleaks.
check_range() {
    local bad big
    [ -n "$(git rev-list "$@")" ] || return 0
    [ -z "$(git rev-list --min-parents=2 "$@")" ] || fail "merge commit in range; keep history linear"
    bad=$(git log --format='%h%x09%an%x09%ae%x09%cn%x09%ce' "$@" \
        | awk -F'\t' -v n="$WANT_NAME" -v e="$WANT_EMAIL" \
            '!($2 == n && $3 == e && $4 == n && $5 == e) { print $1 " " $2 " <" $3 "> / " $4 " <" $5 ">" }')
    [ -z "$bad" ] || fail "wrong identity, expected $WANT_NAME <$WANT_EMAIL>: $(printf '%s\n' "$bad" | head -n 3 | paste -sd';')"
    git log --format=%B "$@" | check_deny "commit messages"
    git log --format=%B "$@" | gitleaks stdin --redact --no-banner --log-level warn >&2 \
        || fail "gitleaks: secret in a commit message"
    git log --format= --name-only --diff-filter=ACMR "$@" | sort -u | check_paths
    big=$(git rev-list --objects "$@" \
        | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
        | awk -v max="$HYGIENE_MAX_BYTES" '$1 == "blob" && $2 > max { print $3 " (" $2 " bytes)" }')
    [ -z "$big" ] || fail "file over 5 MB: $(printf '%s\n' "$big" | head -n 3 | paste -sd';')"
    git log -p -U0 --format= --no-color --no-ext-diff "$@" | added_lines | check_deny "added lines"
    gitleaks git --redact --no-banner --log-level warn --verbose --log-opts="$*" . >&2 \
        || fail "gitleaks: secret in the commits"
}

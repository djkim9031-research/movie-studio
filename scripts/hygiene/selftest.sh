#!/usr/bin/env bash
# Tries what the hooks must refuse, in a throwaway clone that pushes to a
# local bare repository. Planted strings come from the deny list's sample
# lines and the fake token is generated here, so neither is ever committed.
set -uo pipefail
src=$(git rev-parse --show-toplevel)
. "$src/.githooks/lib.sh"
load_deny
want_identity
need_gitleaks
mapfile -t samples < <(sed -n 's/^sample: //p' "$HYGIENE_DENY_FILE")
[ "${#samples[@]}" -gt 0 ] || fail "deny list has no sample lines"
other_email=$(git config --global user.email || true)
if [ -z "$other_email" ] || [ "$other_email" = "$WANT_EMAIL" ]; then
    other_email=someone.else@example.com
fi
token="ghp_$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 36)"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q --bare "$tmp/remote.git"
git clone -q "$src" "$tmp/work"
cd "$tmp/work"
git for-each-ref --format='%(refname)' refs/remotes | xargs -r -n1 git update-ref -d
git remote set-url origin "$tmp/remote.git"
"$src/scripts/hygiene/setup-repo.sh" --name "$WANT_NAME" --email "$WANT_EMAIL" >/dev/null

passed=0 failed=0
row() { printf '%-46s %-28s %s\n' "$1" "$2" "$3"; }
ok() { row "$1" "$2" PASS; passed=$((passed + 1)); }
bad() { row "$1" "$2" "FAIL $3"; failed=$((failed + 1)); }
reset_work() {
    git reset -q --hard origin/main
    git clean -qfdx
    git update-ref -d refs/notes/commits 2>/dev/null || true
}
# expect LABEL TAG CMD...: with TAG "accepted" CMD must succeed; otherwise
# CMD must fail and print TAG.
expect() {
    local label=$1 tag=$2 out rc=0
    shift 2
    out=$("$@" 2>&1) || rc=$?
    if [ "$tag" = accepted ]; then
        if [ "$rc" -eq 0 ]; then ok "$label" "$tag"; else bad "$label" "$tag" "($(tail -n 1 <<< "$out"))"; fi
    elif [ "$rc" -ne 0 ] && grep -qF -- "$tag" <<< "$out"; then
        ok "$label" "refused: $tag"
    else
        bad "$label" "refused: $tag" "(exit $rc)"
    fi
    reset_work
}

add_file() { mkdir -p "$(dirname "$1")" && printf '%s\n' "$2" > "$1" && git add -f -- "$1"; }
commit_file() { add_file "$1" "$2" && git commit -q -m "$3"; }
commit_skip_push() { add_file "$1" "$2" && git commit -q --no-verify -m "$3" && git push -q origin HEAD:main; }
commit_big() { head -c 6291456 /dev/urandom > big.dat && git add big.dat && git commit -q -m "Add big file"; }
commit_other_email() {
    add_file w.txt w && git -c user.email="$other_email" commit -q --no-verify -m "Add w" && git push -q origin HEAD:main
}
push_notes() { git notes add -m note HEAD && git push -q origin refs/notes/commits; }
commit_push_clean() { commit_file clean.txt hello "Add clean file" && git push -q origin HEAD:main; }

row "Attempt" "Expected" "Result"
if out=$(git push -q -u origin HEAD:main 2>&1); then
    ok "push the current history" accepted
else
    bad "push the current history" accepted "($(tail -n 1 <<< "$out"))"
    exit 1
fi
i=0
for s in "${samples[@]}"; do
    i=$((i + 1))
    expect "sample $i in a commit message" "deny-list match" commit_file notes.txt hello "$(printf 'Update notes\n\n%s' "$s")"
    expect "sample $i in a file" "deny-list match" commit_file notes.txt "$s" "Update notes"
done
expect "fake GitHub token in a file" "gitleaks" commit_file config.txt "token = $token" "Add config"
expect "6 MB file" "over 5 MB" commit_big
expect "file under data/, forced past .gitignore" "ignored path" commit_file data/x.txt x "Add data"
expect "other email, committed with --no-verify" "wrong identity" commit_other_email
expect "sample 1 in a message, with --no-verify" "deny-list match" commit_skip_push y.txt y "$(printf 'Add y\n\n%s' "${samples[0]}")"
expect "fake token in a file, with --no-verify" "gitleaks" commit_skip_push config.txt "token = $token" "Add config"
expect "push of a notes record" "notes ref" push_notes
expect "clean commit and push" accepted commit_push_clean
refs=$(git ls-remote "$tmp/remote.git" | awk '{print $2}' | paste -sd' ')
if [ "$refs" = "HEAD refs/heads/main" ]; then ok "refs on the stand-in remote" "HEAD and main only"
else bad "refs on the stand-in remote" "HEAD and main only" "($refs)"; fi

echo
echo "$passed passed, $failed failed"
[ "$failed" -eq 0 ]

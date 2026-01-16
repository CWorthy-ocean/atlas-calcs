#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/CWorthy-ocean/cson-forge"
BRANCH="main"
TARGET_DIR="cson_forge/blueprints"
META_FILE="${TARGET_DIR}/.upstream-meta.json"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to update blueprints." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "${BRANCH}" --filter=blob:none --sparse "${REPO_URL}" "${tmp_dir}"
git -C "${tmp_dir}" sparse-checkout set "cson_forge/blueprints" --quiet

mkdir -p "${TARGET_DIR}"
rsync -a --delete "${tmp_dir}/cson_forge/blueprints/" "${TARGET_DIR}/"

commit_hash="$(git -C "${tmp_dir}" rev-parse HEAD)"
fetched_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "${META_FILE}" <<EOF
{
  "repo_url": "${REPO_URL}",
  "branch": "${BRANCH}",
  "commit": "${commit_hash}",
  "fetched_at_utc": "${fetched_at}"
}
EOF

echo "Updated ${TARGET_DIR} from ${REPO_URL} at ${commit_hash}"

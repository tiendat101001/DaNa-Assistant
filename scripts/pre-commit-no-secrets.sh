#!/usr/bin/env bash
# pre-commit hook: refuse to commit secrets or machine-local config.
# Allow-listed: .env.example (documented template, no real secrets).
set -eu
pattern='(^|/)(\.env(\..+)?|dana\.toml)$'
staged=$(git ls-files --cached | grep -E "$pattern" | grep -v '\.env\.example$' || true)
if [ -n "$staged" ]; then
  echo "Refusing to commit secrets / local config:"
  echo "$staged"
  echo "These are gitignored on purpose — unstage them (git rm --cached <file>)."
  exit 1
fi

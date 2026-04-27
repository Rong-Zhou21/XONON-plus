#!/usr/bin/env bash
set -euo pipefail

message="${1:-}"

if [[ -z "$message" ]]; then
  echo "Usage: bash scripts/git_checkpoint.sh \"commit message\""
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This directory is not a Git repository. Run: git init"
  exit 2
fi

git status --short
git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

git commit -m "$message"

current_branch="$(git branch --show-current)"
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin "$current_branch"
else
  echo "No origin remote configured. Commit was created locally."
fi

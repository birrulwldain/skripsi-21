#!/bin/bash

# Contoh penggunaan: ./assign_issues.sh 5 23

REPO="birrulwldain/skripsi-21"
PROJECT_NUMBER=2
OWNER="@me"
USERNAME=$(gh api user --jq .login)  # Mendapatkan username kamu dari GitHub CLI

START=$1
END=$2

if [[ -z "$START" || -z "$END" ]]; then
  echo "❌ Usage: $0 <start_issue_number> <end_issue_number>"
  exit 1
fi

for ((issue_number=START; issue_number<=END; issue_number++)); do
  ISSUE_URL="https://github.com/$REPO/issues/$issue_number"
  echo "📌 Menambahkan issue #$issue_number ke project..."
  gh project item-add $PROJECT_NUMBER --owner $OWNER --url "$ISSUE_URL"

  echo "👤 Meng-assign issue #$issue_number ke $USERNAME..."
  gh issue edit $issue_number --repo $REPO --add-assignee "$USERNAME"
done

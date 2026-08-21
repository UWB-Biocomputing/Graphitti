#!/usr/bin/env bash
# ==============================================================================
# Graphitti - Stale Issue Cleanup Script
#
# Description:
#   Scans merged pull requests for referenced issue numbers (e.g. '[issue-123]',
#   'fixes #123', 'closes #123', 'issue-123') and checks whether those issues
#   are still in the OPEN state on GitHub.
#
#   In dry-run mode (default), it reports all candidate issues that can be closed.
#   In execute mode (--execute / -x), it closes open issues with a comment linking
#   the merged pull request that resolved them.
#
# Prerequisites:
#   - GitHub CLI ('gh') installed and authenticated ('gh auth login')
#   - 'jq' or 'python3' for JSON processing
#
# Usage:
#   ./.github/scripts/cleanup_stale_issues.sh [OPTIONS]
#
# Options:
#   -d, --dry-run     Preview candidate issues without closing them (default)
#   -x, --execute     Close open issues associated with merged PRs
#   -l, --limit NUM   Number of merged pull requests to inspect (default: 100)
#   -h, --help        Display this help message
# ==============================================================================

set -euo pipefail

DRY_RUN=true
LIMIT=100

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Scan merged pull requests on GitHub and close stale open issues.

Options:
  -d, --dry-run     Preview open issues to close without modifying anything (default)
  -x, --execute     Close the identified issues on GitHub
  -l, --limit NUM   Maximum number of merged PRs to inspect (default: 100)
  -h, --help        Show this help message

Examples:
  $(basename "$0") --dry-run
  $(basename "$0") --execute --limit 200
EOF
  exit 0
}

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -x|--execute)
      DRY_RUN=false
      shift
      ;;
    -l|--limit)
      LIMIT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      usage
      ;;
  esac
done

# Verify GitHub CLI is available and authenticated
if ! command -v gh &> /dev/null; then
  echo "Error: 'gh' (GitHub CLI) is not installed or not in PATH." >&2
  echo "Please install it: https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status &> /dev/null; then
  echo "Error: 'gh' is not authenticated. Please run 'gh auth login' first." >&2
  exit 1
fi

echo "============================================================"
echo " Graphitti Stale Issue Cleanup"
echo " Mode: $( [ "$DRY_RUN" = true ] && echo "DRY RUN (preview only)" || echo "EXECUTE (closing stale issues)" )"
echo " Merged PR scan limit: $LIMIT"
echo "============================================================"
echo ""

echo "Fetching merged pull requests from GitHub..."
PRS_JSON=$(gh pr list --state merged --limit "$LIMIT" --json number,title,body,url,headRefName)

# Extract issue numbers and map them to the merged PRs
# Searches for patterns:
# 1. [issue-123] or [ISSUE-123]
# 2. issue-123 or issue/123 in branch name or text
# 3. (fixes|closes|resolves|closed|fixed|resolved) #123
# 4. #123 in PR title

python3 - "$PRS_JSON" "$DRY_RUN" << 'EOF'
import sys
import json
import re
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed

prs_json_str = sys.argv[1]
dry_run = (sys.argv[2] == "True" or sys.argv[2] == "true")

prs = json.loads(prs_json_str)

issue_patterns = [
    re.compile(r'\[issue[-_](\d+)\]', re.IGNORECASE),
    re.compile(r'\bissue[-_/](\d+)\b', re.IGNORECASE),
    re.compile(r'(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+#(\d+)', re.IGNORECASE)
]

# Map issue_number -> list of PR info dicts
issue_to_prs = {}

for pr in prs:
    pr_num = pr.get("number")
    pr_title = pr.get("title", "")
    pr_body = pr.get("body", "") or ""
    pr_branch = pr.get("headRefName", "") or ""
    pr_url = pr.get("url", "")

    search_texts = [pr_title, pr_branch, pr_body]
    found_issues = set()

    for text in search_texts:
        for pattern in issue_patterns:
            for match in pattern.finditer(text):
                issue_id = int(match.group(1))
                # Skip self-referencing PR number if matched via #<num>
                if issue_id != pr_num:
                    found_issues.add(issue_id)
                #end if
            #end for match
        #end for pattern
    #end for text

    for issue_id in found_issues:
        if issue_id not in issue_to_prs:
            issue_to_prs[issue_id] = []
        #end if
        issue_to_prs[issue_id].append({
            "pr_number": pr_num,
            "pr_title": pr_title,
            "pr_url": pr_url
        })
    #end for issue_id
#end for pr

sorted_issue_ids = sorted(issue_to_prs.keys())
print(f"Discovered {len(sorted_issue_ids)} referenced issue candidates in merged PRs.\n")
print(f"Checking current status of {len(sorted_issue_ids)} candidate issues on GitHub: ", end="", flush=True)

def check_issue_status(issue_id):
    cmd = ["gh", "issue", "view", str(issue_id), "--json", "number,title,state,url"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return (issue_id, None)
    #end if
    try:
        issue_data = json.loads(result.stdout)
        return (issue_id, issue_data)
    except Exception:
        return (issue_id, None)
    #end try
#end def check_issue_status

open_issues_found = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(check_issue_status, i_id): i_id for i_id in sorted_issue_ids}
    for future in as_completed(futures):
        print(".", end="", flush=True)
        issue_id, issue_data = future.result()
        if issue_data and issue_data.get("state") == "OPEN":
            linked_prs = issue_to_prs[issue_id]
            open_issues_found.append((issue_data, linked_prs))
        #end if
    #end for future
#end with

# Sort open issues by issue number for deterministic output
open_issues_found.sort(key=lambda item: item[0]['number'])
print("\n")

if not open_issues_found:
    print("No open stale issues found. All referenced issues in scanned merged PRs are closed.")
    sys.exit(0)
#end if

print(f"Found {len(open_issues_found)} OPEN issues associated with merged pull requests:\n")

for issue_data, linked_prs in open_issues_found:
    issue_num = issue_data['number']
    issue_title = issue_data['title']
    issue_url = issue_data['url']
    pr_references = ", ".join([f"PR #{p['pr_number']} ({p['pr_url']})" for p in linked_prs])

    print(f"  - Issue #{issue_num}: \"{issue_title}\"")
    print(f"    URL: {issue_url}")
    print(f"    Resolved by: {pr_references}")

    if not dry_run:
        comment_msg = f"Closed automatically by cleanup script: resolved in merged pull request {linked_prs[0]['pr_url']}."
        close_cmd = ["gh", "issue", "close", str(issue_num), "--comment", comment_msg]
        close_result = subprocess.run(close_cmd, capture_output=True, text=True)
        if close_result.returncode == 0:
            print(f"    -> Successfully closed Issue #{issue_num}.")
        else:
            print(f"    -> Failed to close Issue #{issue_num}: {close_result.stderr.strip()}")
        #end if
    #end if
    print("")
#end for

if dry_run:
    print("------------------------------------------------------------")
    print(f"DRY RUN COMPLETE: {len(open_issues_found)} issues identified.")
    print("Run with '--execute' to close these issues.")
    print("------------------------------------------------------------")
else:
    print("------------------------------------------------------------")
    print(f"CLEANUP COMPLETE: Processed {len(open_issues_found)} issues.")
    print("------------------------------------------------------------")
#end if
EOF

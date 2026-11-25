"""Publish structured findings as inline comments, with unanchored findings in the body."""

import argparse
import json
import re
import subprocess
from pathlib import Path

TITLE = re.compile(r"^\[(P[0-3]|Requires human review|Suggestion)\] \S")
HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def diff_lines(files):
    """Use API filenames and patch hunks, including context and deleted lines."""
    anchors = set()
    for file in files:
        path = file["filename"]
        left = right = None
        for text in (file.get("patch") or "").splitlines():
            match = HUNK.match(text)
            if match:
                left, right = map(int, match.groups())
            elif left is not None and text.startswith(" "):
                # GitHub requires RIGHT for unchanged context, LEFT for deletions.
                anchors.add((path, "RIGHT", right))
                left += 1
                right += 1
            elif left is not None and text.startswith("-"):
                anchors.add((path, "LEFT", left))
                left += 1
            elif right is not None and text.startswith("+"):
                anchors.add((path, "RIGHT", right))
                right += 1
    return anchors


def review_payload(report, files, commit, run_url, has_merge_ref=True, history=None):
    if not isinstance(report, dict) or set(report) != {"findings"}:
        raise ValueError("Expected an object containing only findings")
    if not isinstance(report["findings"], list):
        raise ValueError("findings must be a list")
    history = [] if history is None else history
    if not isinstance(history, list) or any(
        not isinstance(comment, dict)
        or (comment.get("body") is not None and not isinstance(comment["body"], str))
        for comment in history
    ):
        raise ValueError("history must be a list of comments")
    # Historical issue identity is decided by the reviewer, not by text matching.
    # Only collapse identical records within this one output batch.
    seen = set()
    anchors = diff_lines(files)
    comments, overall = [], []
    for finding in report["findings"]:
        if not isinstance(finding, dict) or set(finding) - {
            "title",
            "body",
            "path",
            "line",
            "side",
        }:
            raise ValueError("Invalid finding fields")
        title, body = finding.get("title"), finding.get("body")
        if (
            not isinstance(title, str)
            or not TITLE.match(title)
            or "\n" in title
            or not isinstance(body, str)
            or not body.strip()
        ):
            raise ValueError("Each finding needs a labeled title and nonempty body")
        path, line = finding.get("path"), finding.get("line")
        side = finding.get("side", "RIGHT")
        if (
            (path is not None and (not isinstance(path, str) or not path))
            or (line is not None and (type(line) is not int or line < 1))
            or side not in ("LEFT", "RIGHT")
        ):
            raise ValueError("Invalid finding location")
        key = (path, side, line, title, body.strip())
        if key in seen:
            continue
        seen.add(key)
        text = "### {}\n\n{}".format(title, body.strip())
        if (path, side, line) in anchors:
            comments.append(dict(path=path, line=line, side=side, body=text))
        else:
            # Binary/truncated patches and out-of-diff lines remain visible.
            location = ""
            if path:
                location = "\n\n`{}`{}".format(
                    path, ":{} ({})".format(line, side) if line else ""
                )
            overall.append(text + location)
        if len(text.encode("utf-8")) > 60000:
            raise ValueError("Finding exceeds the comment size limit")
    if not comments and not overall and history:
        return None
    if not comments and not overall:
        overall.append("No actionable findings.")
    if not has_merge_ref:
        overall.append("No merge ref was available; reviewed the PR head only.")
    overall.append("_AI-generated; not an approval. [Run]({})._".format(run_url))
    body = "\n\n".join(overall)
    if len(body.encode("utf-8")) > 60000:
        raise ValueError("Overall review exceeds the comment size limit")
    return dict(commit_id=commit, event="COMMENT", body=body, comments=comments)


def gh_api(endpoint, payload=None):
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    return json.loads(
        subprocess.check_output(
            command,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--files", required=True, type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()
    payload = review_payload(
        json.loads(args.report.read_text()),
        json.loads(args.files.read_text()),
        args.commit,
        args.run_url,
        has_merge_ref=not args.head_only,
        history=json.loads(args.history.read_text()) if args.history else [],
    )
    if payload is None:
        print("No new findings; no review posted.")
        return
    endpoint = "repos/{}/pulls/{}".format(args.repo, args.pr)
    if gh_api(endpoint)["head"]["sha"] != args.commit:
        raise RuntimeError(
            "PR head changed during review; rerun instead of posting stale findings"
        )
    result = gh_api(endpoint + "/reviews", payload)
    print("Published review: " + result["html_url"])


if __name__ == "__main__":
    main()

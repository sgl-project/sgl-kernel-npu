#!/bin/bash
# Push Instructions for feature/add-three-functions branch
# Execute in environment with direct GitHub access

echo "=== Branch Information ==="
echo "Branch: feature/add-three-functions"
echo "Commits: 4 (English messages)"
echo

git log --oneline --decorate | head -5
echo

echo "=== Please enter GitHub Token (with repo permission): ==="
read -s TOKEN

git remote set-url origin https://${TOKEN}@github.com/1329009851/sgl-kernel-npu.git

echo "=== Pushing to GitHub ==="
git push -u origin feature/add-three-functions

if [ $? -eq 0 ]; then
    echo
    echo "✅ Success!"
    echo "Branch: https://github.com/1329009851/sgl-kernel-npu/tree/feature/add-three-functions"
    echo "Create PR: https://github.com/1329009851/sgl-kernel-npu/pull/new/feature/add-three-functions"
fi

# Clean token
git remote set-url origin https://github.com/1329009851/sgl-kernel-npu.git

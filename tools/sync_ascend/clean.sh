#!/bin/bash

# 清理同步代码，恢复到同步之前
#
# 用法:
#   bash tools/sync_triton_ascend.sh [BASE_COMMIT]

set -e

git stash
git clean -xdf bin/
git clean -xdf include/
git clean -xdf lib/
git clean -xdf python/
git clean -xdf test/
git clean -xdf third_party/ascend/

pushd third_party/flir/
git stash
git clean -xdf include/
git clean -xdf lib/
popd

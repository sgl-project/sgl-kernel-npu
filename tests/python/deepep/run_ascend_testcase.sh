test_case=$1

LOCAL_BUILD_DIR=/tmp/sglang_deepep_build
rm -rf $LOCAL_BUILD_DIR
mkdir -p $LOCAL_BUILD_DIR
chmod -R 777 $LOCAL_BUILD_DIR

echo "============================================= Copy code to local directory ============================================"
echo "Source path: $SGLANG_SOURCE_PATH"
echo "Local build path: $LOCAL_BUILD_DIR"
cp -r $SGLANG_SOURCE_PATH/* $LOCAL_BUILD_DIR/

cd $LOCAL_BUILD_DIR

if [ ! -f "${test_case}" ];then
  echo "ERROR: The test case file is not exist: $LOCAL_BUILD_DIR/${test_case}"
  ls -l $(dirname "${test_case}")
  exit 1
fi

# speed up by using infra cache services
CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
pip config set global.index-url http://${CACHING_URL}/pypi/simple
pip config set global.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
pip config set global.trusted-host "${CACHING_URL} pypi.tuna.tsinghua.edu.cn"

pip3 install kubernetes
pip3 install xgrammar==0.1.25

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

export WORLD_SIZE=2
export HCCL_BUFFSIZE=3000
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "============================================= Start building dependency ============================================="
bash scripts/npu_ci_install_dependency.sh
echo "============================================= Finished building dependency ============================================="


echo "============================================= Start building deep-ep ============================================="
pkill -9 cmake 2>/dev/null || true
pkill -9 gmake 2>/dev/null || true
pkill -9 make 2>/dev/null || true
sleep 3

find ./output -type d -exec chmod 755 {} \; 2>/dev/null || true
find ./build -type d -exec chmod 755 {} \; 2>/dev/null || true
find ./csrc/deepep/ops2/build_out -type d -exec chmod 755 {} \; 2>/dev/null || true

rm -rf ./output 2>/dev/null || true
rm -rf ./build 2>/dev/null || true
rm -rf ./csrc/deepep/ops2/build_out 2>/dev/null || true

find . -type d -name "CMakeFiles" -exec chmod -R 755 {} \; 2>/dev/null || true
find . -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "cmake.check_cache" -delete 2>/dev/null || true

max_retries=3
attempt=0
build_success=false
while [ $attempt -lt $max_retries ]; do
    echo "Attempt $((attempt+1)) of $max_retries..."
    cd "$LOCAL_BUILD_DIR"
    find ./output -type d -exec chmod 755 {} \; 2>/dev/null || true
    find ./build -type d -exec chmod 755 {} \; 2>/dev/null || true
    find ./csrc/deepep/ops2/build_out -type d -exec chmod 755 {} \; 2>/dev/null || true
    rm -rf ./output ./build ./csrc/deepep/ops2/build_out 2>/dev/null || true
    sleep 1

    bash build.sh -a deepep2
    if ls ./output/deep_ep*.whl 1> /dev/null 2>&1; then
        echo "Build successful! Whl file found in local directory: $LOCAL_BUILD_DIR/output"
        build_success=true
        break
    else
        echo "Build failed. Cleaning up local directory..."
        find ./output -type d -exec chmod 755 {} \; 2>/dev/null || true
        find ./build -type d -exec chmod 755 {} \; 2>/dev/null || true
        find ./csrc/deepep/ops2/build_out -type d -exec chmod 755 {} \; 2>/dev/null || true
        rm -rf ./output ./build ./csrc/deepep/ops2/build_out 2>/dev/null || true
        echo "Waiting 10 seconds for file system sync..."
        sleep 10
        attempt=$((attempt+1))
    fi
done

if [ "$build_success" = false ]; then
    echo "ERROR: deep-ep build failed after $max_retries attempts in local directory: $LOCAL_BUILD_DIR"
    ls -l $LOCAL_BUILD_DIR/
    ls -l $LOCAL_BUILD_DIR/output/ 2>/dev/null || true
    exit 1
fi

pip install ./output/deep_ep*.whl --no-cache-dir
cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
ln -sf deep_ep/deep_ep_cpp*.so ./ 2>/dev/null || true
echo "============================================= Finished building deep-ep ============================================="

pip show deep-ep

echo "============================================= Running test case ${test_case} ============================================"
cd $LOCAL_BUILD_DIR
python3 -u ${test_case}
echo "============================================= Finished test case ${test_case} ============================================"

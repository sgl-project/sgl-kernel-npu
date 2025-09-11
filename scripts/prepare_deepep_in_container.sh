#!/bin/bash

while getopts ":a:" opt; do
    case ${opt} in
        a )
            BUILD_ARGS="$OPTARG"
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

cd ${GITHUB_WORKSPACE}
if [ -n "$BUILD_ARGS" ]; then
    bash build.sh -a "$BUILD_ARGS"
else
    bash build.sh
fi
pip install ${GITHUB_WORKSPACE}/output/deep_ep*.whl --no-cache-dir
cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
ln -s deep_ep/deep_ep_cpp*.so || true

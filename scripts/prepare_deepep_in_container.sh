set -e

BUILD_ARGS=""
SOC_VERSION=""
SKIP_BUILD=false

while getopts ":a:s" opt; do
    case ${opt} in
        a )
            BUILD_ARGS="$OPTARG"
            ;;
        s )
            SKIP_BUILD=true
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
SOC_VERSION="${1:-}"

cd ${GITHUB_WORKSPACE}

if [ "$SKIP_BUILD" = false ]; then
    if [ -n "$BUILD_ARGS" ]; then
        if [ -n "$SOC_VERSION" ]; then
            bash build.sh -a "$BUILD_ARGS" "$SOC_VERSION"
        else
            bash build.sh -a "$BUILD_ARGS"
        fi
    else
        bash build.sh
    fi
fi

pip install ${GITHUB_WORKSPACE}/output/deep_ep*.whl --no-cache-dir

DEEP_EP_LOCATION=$(pip show deep-ep | awk '/^Location:/ {print $2}')
if [ -z "$DEEP_EP_LOCATION" ]; then
    echo "Error: deep-ep package not found after pip install" 1>&2
    exit 1
fi
cd "$DEEP_EP_LOCATION"
ln -s deep_ep/deep_ep_cpp*.so || true

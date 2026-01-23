#!/bin/bash
CopyOps() {
    local src_dir="$1"  # Path of Source Directory A
    local dst_dir="$2"  # Path of Target Directory B

    if [ -d "$src_dir/op_host" ]; then
        cp -rf "$src_dir/op_host/"* "$dst_dir/op_host/"
        echo "cp -rf $src_dir/op_host/* $dst_dir/op_host/"
    fi

    if [ -d "$src_dir/op_kernel" ]; then
        cp -rf "$src_dir/op_kernel/"* "$dst_dir/op_kernel/"
        echo "cp -rf $src_dir/op_kernel/* $dst_dir/op_kernel/"
    fi

    # Ensure that op_host and op_kernel exist in the target directory
    mkdir -p "$dst_dir/op_host" "$dst_dir/op_kernel"

    # Traverse all direct subdirectories under the source directory (including directory names with spaces)
    find "$src_dir" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' subdir; do
        # Check whether the subdirectory exists (although find has already filtered it, double-checking is safer)
        if [ -d "$subdir" ]; then
            # Processing the op_host directory
            if [ -d "$subdir/op_host" ]; then
                cp -rf "$subdir/op_host/"* "$dst_dir/op_host/"
                echo "cp -rf $subdir/op_host/* $dst_dir/op_host/"
            fi

            # Processing the op_kernel directory
            if [ -d "$subdir/op_kernel" ]; then
                cp -rf "$subdir/op_kernel/"* "$dst_dir/op_kernel/"
                echo "cp -rf $subdir/op_kernel/* $dst_dir/op_kernel/"
            fi
        fi
    done
    echo "${src_dir} built successfully"
}

BuildOps() {
  local proj_name=$1
  echo "${proj_name}"
  local soc_version=$2

  if [ -d "./${proj_name}" ]; then
    rm -rf ${proj_name}/cmake
    rm -rf ${proj_name}/op_host
    rm -rf ${proj_name}/op_kernel
    rm -rf ${proj_name}/scripts
    rm -rf ${proj_name}/build.sh
    rm -rf ${proj_name}/CMakeLists.txt
    rm -rf ${proj_name}/CMakePresets.json
    rm -rf ${proj_name}/framework
  fi

  echo "msopgen gen -i ./AddCustom.json -c ai_core-${soc_version} -f pytorch -lan cpp -out ${proj_name}"
  msopgen gen -i ./AddCustom.json -c ai_core-${soc_version} -f pytorch -lan cpp -out ${proj_name}
  rm -rf ./${proj_name}/op_host/add_custom*
  rm -rf ./${proj_name}/op_kernel/add_custom*
}

# Build the operator project and transfer its output to the specified location
BuildAscendProj() {
  local os_id=$(grep ^ID= /etc/os-release | cut -d= -f2 | tr -d '"')
  local arch=$(uname -m)
  local soc_version=$2
  local is_extract=$3
  local build_type=$4
  # Modify the default operator name
  export OPS_PROJECT_NAME=aclnnInner
  # Enter the compilation path
  echo "cd $1"
  cd $1

  BuildOps "ops_${soc_version}" ${soc_version}
  BuildOps "ops2_${soc_version}" ${soc_version}
  CopyOps "./ops" "./ops_${soc_version}"
  CopyOps "./ops2" "./ops2_${soc_version}"
  cp -r ./ops_${soc_version}/cmake ./ops
  cp -r ./ops2_${soc_version}/cmake ./ops2
}

BuildAscendProj $1 $2 $3 $4
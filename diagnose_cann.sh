#!/bin/bash

echo "======================================"
echo "CANN 8.5 Environment Diagnosis"
echo "======================================"

# Check environment variables
echo "Environment Variables:"
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}"
echo "ASCEND_AICPU_PATH: ${ASCEND_AICPU_PATH}"
echo ""

# Find ascendc_kernel_cmake directory
echo "Searching for ascendc_kernel_cmake directory..."
if [ -n "$ASCEND_HOME_PATH" ]; then
    echo "Checking paths under ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
    
    # Check possible locations
    PATH1="$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake"
    PATH2="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake"
    PATH3="$ASCEND_HOME_PATH/ascendc_devkit/tikcpp/samples/cmake"
    
    for path in "$PATH1" "$PATH2" "$PATH3"; do
        if [ -d "$path" ]; then
            echo "✓ Found: $path"
            echo "  Contents:"
            ls -la "$path" 2>/dev/null | head -20
            
            # Search for ASC config files
            echo ""
            echo "  Searching for ASC config files..."
            find "$path" -name "ASCConfig.cmake" -o -name "asc-config.cmake" 2>/dev/null
            
            # Search for any cmake config files
            echo ""
            echo "  All cmake config files in this directory:"
            find "$path" -name "*.cmake" -type f 2>/dev/null | head -10
        else
            echo "✗ Not found: $path"
        fi
        echo ""
    done
else
    echo "ASCEND_HOME_PATH is not set!"
fi

echo "======================================"
echo "Searching entire CANN installation..."
echo "======================================"

if [ -n "$ASCEND_HOME_PATH" ]; then
    echo "Searching for ASCConfig.cmake..."
    find "$ASCEND_HOME_PATH" -name "ASCConfig.cmake" 2>/dev/null
    
    echo ""
    echo "Searching for asc-config.cmake..."
    find "$ASCEND_HOME_PATH" -name "asc-config.cmake" 2>/dev/null
    
    echo ""
    echo "Searching for ASC.cmake..."
    find "$ASCEND_HOME_PATH" -name "ASC.cmake" 2>/dev/null
fi

echo ""
echo "======================================"
echo "CANN Version Information"
echo "======================================"

# Try to get version
if [ -f "/etc/Ascend/ascend_cann_install.info" ]; then
    echo "Installation info:"
    cat "/etc/Ascend/ascend_cann_install.info"
else
    echo "No installation info file found"
fi

# Check version.info if exists
if [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/version.info" ]; then
    echo ""
    echo "Version info from ASCEND_HOME_PATH:"
    cat "$ASCEND_HOME_PATH/version.info"
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "${ASCEND_HOME_PATH}/../version.info" ]; then
    echo ""
    echo "Version info from parent directory:"
    cat "${ASCEND_HOME_PATH}/../version.info"
fi

echo ""
echo "======================================"
echo "Diagnosis Complete"
echo "======================================"
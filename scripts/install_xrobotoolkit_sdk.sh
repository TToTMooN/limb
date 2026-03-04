#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="$REPO_ROOT/dependencies"

# Check that we're inside a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Error: No virtual environment is currently activated."
    echo "Please activate your venv first: source .venv/bin/activate"
    exit 1
fi

echo "Installing build dependencies (pybind11, cmake, setuptools, wheel)..."
uv pip install pybind11 cmake setuptools wheel

export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "pybind11_DIR=$pybind11_DIR"

SDK_DIR="$DEPS_DIR/XRoboToolkit-PC-Service-Pybind"

if [[ -d "$SDK_DIR" ]]; then
    echo "XRoboToolkit-PC-Service-Pybind already cloned at $SDK_DIR"
else
    echo "Cloning XRoboToolkit-PC-Service-Pybind..."
    cd "$DEPS_DIR"
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
fi

cd "$SDK_DIR"

# Build native SDK library if not already built
if [[ ! -f lib/libPXREARobotSDK.so ]]; then
    echo "Building native SDK library..."
    mkdir -p tmp
    cd tmp
    if [[ ! -d "XRoboToolkit-PC-Service" ]]; then
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
    fi
    cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
    bash build.sh
    cd "$SDK_DIR"

    mkdir -p lib include
    cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
    cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/

    if [[ -f tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so ]]; then
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
    elif [[ -f tmp/XRoboToolkit-PC-Service/RoboticsService/SDK/linux/64/libPXREARobotSDK.so ]]; then
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/SDK/linux/64/libPXREARobotSDK.so lib/
    else
        echo "Error: libPXREARobotSDK.so not found after build"
        exit 1
    fi
    echo "Native SDK library built successfully."
else
    echo "Native SDK library already exists at $SDK_DIR/lib/"
fi

echo "Building and installing xrobotoolkit_sdk Python bindings..."
cd "$SDK_DIR"
uv pip install --no-build-isolation . || { echo "Failed to install xrobotoolkit_sdk"; exit 1; }

echo ""
echo "[INFO] xrobotoolkit_sdk installed successfully."
echo "  Verify with: python -c 'import xrobotoolkit_sdk; print(\"OK\")'"

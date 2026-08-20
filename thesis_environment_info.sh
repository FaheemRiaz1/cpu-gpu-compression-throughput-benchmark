#!/usr/bin/env bash

echo "=================================================="
echo "OPERATING SYSTEM"
echo "=================================================="
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
echo "Kernel: $(uname -r)"

echo
echo "=================================================="
echo "CPU"
echo "=================================================="
lscpu | grep -E 'Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|^CPU\(s\):'

echo
echo "=================================================="
echo "SYSTEM MEMORY"
echo "=================================================="
free -h

echo
echo "=================================================="
echo "GPU AND DRIVER"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id \
  --format=csv,noheader

echo
echo "=================================================="
echo "GPU PCIE INFORMATION"
echo "=================================================="
nvidia-smi -q | grep -A 20 "GPU Link Info" | head -n 25

echo
echo "=================================================="
echo "CUDA TOOLKIT"
echo "=================================================="
nvcc --version 2>/dev/null || echo "nvcc not found"
cat /usr/local/cuda/version.json 2>/dev/null | head -n 20

echo
echo "=================================================="
echo "C++ COMPILER"
echo "=================================================="
g++ --version 2>/dev/null | head -n 1
gcc --version 2>/dev/null | head -n 1

echo
echo "=================================================="
echo "CMAKE"
echo "=================================================="
cmake --version 2>/dev/null | head -n 1

echo
echo "=================================================="
echo "LZ4 VERSION"
echo "=================================================="
pkg-config --modversion liblz4 2>/dev/null \
  || dpkg-query -W -f='${Version}\n' liblz4-dev 2>/dev/null \
  || echo "LZ4 version not detected automatically"

echo
echo "=================================================="
echo "NVCOMP INSTALLATION"
echo "=================================================="
source ~/nvcomp_env/bin/activate 2>/dev/null
python -m pip list 2>/dev/null | grep -i nvcomp \
  || echo "No nvCOMP Python package name found"

find ~/nvcomp_env -maxdepth 6 \
  \( -iname '*nvcomp*.so*' -o -iname 'nvcomp.h' -o -iname '*nvcomp*version*' \) \
  2>/dev/null | head -n 20

echo
echo "nvCOMP version macros:"
find ~/nvcomp_env -maxdepth 7 -type f \
  \( -name '*.h' -o -name '*.hpp' \) 2>/dev/null \
  -exec grep -H -E \
  'NVCOMP_VERSION|NVCOMP_MAJOR_VERSION|NVCOMP_MINOR_VERSION|NVCOMP_PATCH_VERSION' \
  {} \; | head -n 20

echo
echo "=================================================="
echo "STORAGE"
echo "=================================================="
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN,TYPE

echo
echo "=================================================="
echo "COMPILATION COMMANDS AND FLAGS"
echo "=================================================="
grep -RInE \
  'nvcc|g\+\+|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS|-O[0-3]|--use_fast_math|-std=c\+\+' \
  scripts CMakeLists.txt Makefile 2>/dev/null | head -n 120

echo
echo "=================================================="
echo "PROJECT LOCATION"
echo "=================================================="
pwd

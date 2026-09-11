#!/usr/bin/env bash
set -e

cd ~/gpu_benchmark_clean

echo "Building FastLanes SPJA x40 co-processor benchmark..."

SRC="external/baseline/fastlanes_gpu/spja_coproc_fastlanes.cu"
OBJ="build/baseline/fastlanes_gpu/CMakeFiles/fastlanes_gpu_aggregate.dir/aggregate.cu.o"
INCLUDES="build/baseline/fastlanes_gpu/CMakeFiles/fastlanes_gpu_aggregate.dir/includes_CUDA.rsp"

/usr/bin/nvcc \
-DBOOST_PROGRAM_OPTIONS_DYN_LINK -DBOOST_PROGRAM_OPTIONS_NO_LIB -DCASDEC_ENABLE_OP_FUSION -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
$(cat "$INCLUDES") \
-DFMT_USE_NONTYPE_TEMPLATE_ARGS=0 --expt-relaxed-constexpr --expt-relaxed-constexpr \
--ptxas-options=--warn-on-double-precision-use,--warn-on-spills,--warning-as-error \
--ptxas-options=--warn-on-local-memory-usage \
--generate-line-info -std=c++20 -arch=native -Xcompiler=-fPIE \
-c "$SRC" \
-o "$OBJ"

cd build/baseline/fastlanes_gpu
bash CMakeFiles/fastlanes_gpu_aggregate.dir/link.txt
cd ~/gpu_benchmark_clean

echo "Running FastLanes SPJA x40 co-processor benchmark..."

./build/baseline/fastlanes_gpu/fastlanes_gpu_aggregate

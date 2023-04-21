#!/bin/bash

rootdir=/cuda-graph-analytics/ldbc_graphalytics_platforms_graphblas/graphalytics-1.10.0-graphblas-0.1.0-SNAPSHOT
cd ${rootdir}
# remove old code and config
rm -rf ./bin/exe
rm -rf ./src/main/c
rm ./config/benchmark.properties
# copy new code and config
cp -r /src ./src/main/c
cp /src/benchmark-configs/benchmark.properties ./config/
cp /src/benchmark-configs/cdlp.properties ./config/
# build C and CUDA code
bin/sh/build-wrapper-only.sh
# run benchmark
bin/sh/run-benchmark.sh
# copy report to build directory and get feedback from rai
cp -r ./report/ /build
#!/bin/bash

set -euo pipefail

# download dataset if you want to test a different dataset not already in the docker image
# FAILURE: rai does not support downloading from internet, so you need to download the dataset and copy it to the docker image
# datasetdir=/cuda-graph-analytics/example-data-sets/
# wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_9-fb.tar.zst | unzstd | tar -xvf - -C $datasetdir/graphs

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
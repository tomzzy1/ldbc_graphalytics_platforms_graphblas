#/bin/bash

# set -eo pipefail

rootdir="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" >/dev/null 2>&1 && pwd )/.."
cd ${rootdir}
. scripts/project-vars.sh

rm -rf cd ${PROJECT}/src/main/c
cp -r src/main/c ${PROJECT}/src/main/c

cd ${PROJECT}
rm -rf bin/exe
bin/sh/build-wrapper-only.sh

# directly run the benchmarks
bin/sh/run-benchmark.sh

#!/bin/bash

rootdir="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" >/dev/null 2>&1 && pwd )/.."
cd ${rootdir}


cp -r $rootdir/small-data-sets/graphs/datagen-7_5-fb-CDLP /cuda-graph-analytics/example-data-sets/graphs
cp -r $rootdir/small-data-sets/graphs/datagen-7_5-fb.* /cuda-graph-analytics/example-data-sets/graphs
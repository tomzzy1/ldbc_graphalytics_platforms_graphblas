#!/bin/bash

rootdir="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" >/dev/null 2>&1 && pwd )/.."
cd ${rootdir}

cp -r $rootdir/small-data-sets/graphs/cit* /cuda-graph-analytics/example-data-sets/graphs
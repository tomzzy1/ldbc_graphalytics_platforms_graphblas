#!/bin/bash

set -euo pipefail

apt-get install zstd

# create subdir ./graphs if not exists
rootdir="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" >/dev/null 2>&1 && pwd )/.."
cd ${rootdir}
echo $rootdir
[ ! -d $rootdir/small-data-sets/graphs ] && mkdir $rootdir/small-data-sets/graphs

wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/cit-Patents.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_5-fb.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_6-fb.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_7-zf.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_8-zf.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_9-fb.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/dota-league.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/graph500-22.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/kgs.tar.zst  | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs
wget -qO- https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/wiki-Talk.tar.zst | unzstd | tar -xvf - -C $rootdir/small-data-sets/graphs


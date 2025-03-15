#!/bin/bash

# Install required tools if needed
# sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`

# Clone FlameGraph if needed

# Record performance data
echo "Recording performance data..."
sudo perf record -F 99 -g -a -- cmake-build-debug/fts -d data/guthenberg/pip -a inverted -s tf-idf


FLAMEGRAPH_DIR="/home/dhia/Downloads/FlameGraph"


# Generate flamegraph
echo "Generating flamegraph..."
sudo perf script > out.perf
$FLAMEGRAPH_DIR/stackcollapse-perf.pl out.perf > out.folded
$FLAMEGRAPH_DIR/flamegraph.pl out.folded > flamegraph.svg

echo "Flamegraph generated as flamegraph.svg"
echo "Open it in your web browser to view"
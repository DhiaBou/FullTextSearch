#!/bin/bash

# Install required tools if needed
# sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`

# Record performance data
echo "Recording performance data..."
sudo perf record -F 99 -g -a -- ../cmake-build-debug/fts -d ../data/ppip -a inverted -s tf-idf -b

FLAMEGRAPH_DIR=$HOME/Downloads/FlameGraph

# Generate flamegraph
echo "Generating flamegraph..."

# Get current date and time for the SVG file name
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

sudo perf script > out.perf
$FLAMEGRAPH_DIR/stackcollapse-perf.pl out.perf > out.folded
$FLAMEGRAPH_DIR/flamegraph.pl out.folded > "flamegraph_$TIMESTAMP.svg"
rm out.perf out.folded
rm -f perf.data
rm -f perf.data.old

echo "Flamegraph generated as flamegraph_$TIMESTAMP.svg"
echo "Open it in your web browser to view"

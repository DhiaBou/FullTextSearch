import os
import subprocess
import csv
import argparse
from datetime import datetime
import subprocess

# -------------- Setup --------------

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Benchmark script for FullTextSearch.")
parser.add_argument('-d', '--data', required=True, help="Path to the data directory.")
parser.add_argument('-e', '--executable', required=True, help="Path to the FullTextSearch executable.")
parser.add_argument('-o', '--output', required=True, help="Path to the output directory.")
parser.add_argument('-q', '--query', required=True, help="Path to the query directory.")
parser.add_argument('-n', '--num_results', required=True, help="The number of results per query.")
args = parser.parse_args()

data_src = args.data
executable_path = args.executable
output_dir = args.output
query_dir = args.query
num_results = args.num_results
data_sets = ['imdb', 'yelp', 'cnn_dailymail']
algos = ['inverted', 'trigram']

# Number of iterations
iterations = 5

# Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

# -------------- Benchmarks --------------

# BUILD
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
build_file = f'{output_dir}/build_{timestamp}.csv'
footprint_file = f'{output_dir}/footprint_{timestamp}.csv'
query_file = f'{output_dir}/query_{timestamp}.csv'
with open(build_file, 'w', newline='') as build,\
    open(footprint_file, 'w', newline='') as footprint,\
    open(query_file, 'w', newline='') as query:

    build_writer = csv.writer(build)
    footprint_writer = csv.writer(footprint)
    query_writer = csv.writer(query)
    
    for i in range(iterations):
        for algo in algos:
            subprocess.run(["sudo", "purge"])
             
            for data in data_sets:
                print(f"{algo} on {data}")
                result = subprocess.run([
                    executable_path,
                    '-d', os.path.join(data_src, data),
                    '-a', algo,
                    '-s', 'bm25',
                    '-n', num_results,
                    '-q', os.path.join(query_dir, data, "performance_queries"),
                    '-o', algo
                ], capture_output=True, text=True)

                for line in result.stdout.splitlines():
                    if line.startswith("Build:"):
                        build_time = line.split(":")[1].strip()
                        build_writer.writerow([data, algo, build_time])
                    elif line.startswith("Real memory footprint:"):
                        memory_footprint = line.split(":")[1].strip()
                        footprint_writer.writerow([data, algo, "size", memory_footprint])
                    elif line.startswith("Allocated memory footprint:"):
                        memory_footprint = line.split(":")[1].strip()
                        footprint_writer.writerow([data, algo, "capacity", memory_footprint])
                    else:
                        query_id = line.split(":")[0].strip()  # Remove quotes from the query
                        time = line.split(":")[1].strip()  # Extract the time
                        query_writer.writerow([data, algo, query_id, time])

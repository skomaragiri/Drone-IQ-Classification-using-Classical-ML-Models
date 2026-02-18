#!/bin/bash

# Usage: ./convert_sigmf.sh /path/to/input_dir /path/to/output_dir

input_dir="$1"
output_dir="$2"

# Check arguments
if [[ -z "$input_dir" || -z "$output_dir" ]]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Make sure output directory exists
mkdir -p "$output_dir"

# Find all .sigmf-meta files recursively and convert
find "$input_dir" -type f -name "*.sigmf-meta" | while read -r f; do
    out_file="$output_dir/$(basename "${f%.*}").csv"
    python3 conv_sigmf_to_iq_csv.py --meta "$f" --out "$out_file"
done

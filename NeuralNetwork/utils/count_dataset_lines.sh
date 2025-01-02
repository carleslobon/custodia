#!/bin/bash

# File path
FILE_PATH="/mnt/c/Users/mikig/Desktop/UPC/PAE/Datasets/9810e03bba4983da_MOHANAD_A4706/9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv"

# Count lines excluding the header
LINE_COUNT=$(($(wc -l < "$FILE_PATH") - 1))

echo "Number of lines (excluding header): $LINE_COUNT"


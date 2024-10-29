#!/bin/bash

# Check if the Python script exists
if [ ! -f backend_inlinemetric.py ]; then
    echo "Python script 'backend_inlinemetric.py' not found!"
    exit 1
fi

# Arrays of batch sizes and number of cores to iterate through
batch_sizes=(2 4 8) #(1 2 4 8)
core_counts=(8 14) #(1 2 4 8 14)
# batch_sizes=(2)
# core_counts=(4 8)

# Output file for the results
output_file="results.txt"

# Clear the output file or create it if it doesn't exist
> $output_file

# Iterate over batch sizes and core counts
for batch_size in "${batch_sizes[@]}"; do
    for num_cores in "${core_counts[@]}"; do
        
        echo "Running with batch size $batch_size and $num_cores cores" | tee -a $output_file
        
        # Set the number of threads for PyTorch and related libraries
        export OMP_NUM_THREADS=$num_cores
        export MKL_NUM_THREADS=$num_cores

        # Run the Python script with the batch size and core count
        python3 backend_inlinemetric.py $num_cores $batch_size | tee -a $output_file

        echo -e "\n-----------------------------------\n" | tee -a $output_file

    done
done

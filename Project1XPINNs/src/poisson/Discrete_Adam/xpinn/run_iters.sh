#!/bin/zsh

# Define the number of iterations
ITERATIONS=10

# Loop for the specified number of iterations
for ((i=5; i<=$ITERATIONS; i++)); do
    echo "Running iteration $i"

    # Run a.py
    python3 PINN_points.py

    # Run b.py with the iteration number as an argument
    python3 PINN_experiment.py $i
done

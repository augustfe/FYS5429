#!/bin/zsh

# Define the number of iterations
ITERATIONS=10

echo "Making test points"
python3 PINN_test_points.py

# Loop for the specified number of iterations
for ((i=1; i<=$ITERATIONS; i++)); do
    echo "Running iteration $i"

    # Run a.py
    python3 PINN_points.py

    # Run b.py with the iteration number as an argument
    python3 PINN_experiment.py $i
done

#!/bin/zsh

# Define the number of iterations
ITERATIONS=10

echo "Making test points"
python3 XPINN_test_points.py

# Loop for the specified number of iterations
for ((i=1; i<=$ITERATIONS; i++)); do
    echo "Running iteration $i"

    # Run a.py
    python3 XPINN_points.py

    # Run b.py with the iteration number as an argument
    python3 XPINN_experiment.py $i
done

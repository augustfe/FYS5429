#!/bin/zsh

# Define the number of iterations
python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 1 1 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 20 1 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 20 20 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 40 40 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 20 40 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 40 20 1

python3 laminar_experiment.py /Users/junmiaohu/Documents/UiO/FYS5429/FYS5429/Project1XPINNs/data/NavierStokes/laminar_decomp_train_400_2100_v2.json 20 80 1
# Project 1 - Extended Physics-Informed Neural Networks <!-- omit in toc -->

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  
## Introduction

Extended Physics-Informed Neural Networks (XPINNs) is a project aimed at extending Physics-Informed Neural Networks through subdomain decomposition. The goal is to improve the predictive accuracy of these networks, while decreasing the required complexity of the model through dividing the workload across multiple individual internal subnetworks. The hope is that this can produce viable results, especially in scenarios where data is sparse or expensive to generate, while offering new oppurtunities for parallelization by dividing the total workload. PINNs encode the underlying physics of the problem by hand directly into the network, guiding the learning process and providing more feasible results, while removing the necessity of prior training data. In implementing XPINNs, we additionally enforce interface conditions between the networks, in order to ensure continuity across subdomain boundaries.

In this project, we provide a package to simplify the process of setting network for training, as well as a setup for generating the relevant subdomain points. This allows us to easily test multiple problems, from the simple Advection or Transport equation to more complex Navier-Stokes flow. We utilize Python3 in combination with JAX in order to ensure the performance of the computations, while keeping the interface user-friendly.

## Features

We provide a simple interface for setting up PINNs, wherin all the user has to do is implement the loss functions for the problem, and input a JSON-file containing the subdomain decomposition. We also provide a setup for generating the decomposition in the correct format.

## Installation

The package is setup through calling
```bash
$ pip3 install .
```
in the Project1XPINNs directory.

## Usage

We provide Jupyter Notebooks illustrating the usage of the package, e.g. [main_advection.ipynb](src/advection/main_advection.ipynb).

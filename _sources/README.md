# Training Pointer Networks Better via Evolutionary Algorithms

## Introduction

This repository contains the implementation of the paper **"Training Pointer Networks Better via Evolutionary Algorithms"**. The paper explores the use of evolutionary algorithms to enhance the training process of Pointer Networks, a type of neural network architecture designed for solving combinatorial optimization problems.

## Abstract

Pointer Networks have shown great promise in solving various combinatorial optimization problems. However, their training can be challenging due to the complexity of the tasks they are designed to solve. This work proposes a novel approach to improve the training of Pointer Networks using evolutionary algorithms. By leveraging the strengths of evolutionary strategies, we aim to achieve better performance and generalization in Pointer Networks.

## Repository Structure

- `src/`: Contains the source code for the implementation.
- `data/`: Includes datasets used for training and evaluation.
- `experiments/`: Scripts and configurations for running experiments.
- `results/`: Stores the results of the experiments.
- `docs/`: Documentation and additional resources.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the Pointer Network using evolutionary algorithms, run the following command:

```sh
python src/train.py --config configs/evolutionary_config.json
# Neural Decoding With Optimization of Node Activations

This repository contains an implementation of the neural decoding technique described in the paper "Neural Decoding With Optimization of Node Activations" by Eliya Nachmani and Yair Be'ery. The paper introduces two novel loss terms to improve the performance of neural decoders for error-correcting codes.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The problem of maximum likelihood decoding with a neural decoder for error-correcting code is considered. The neural decoder can be improved with two novel loss terms on the node’s activations:
1. Sparse Constraint Loss
2. Knowledge Distillation Loss

The proposed method has the same run-time complexity and model size as the neural Belief Propagation decoder while improving the decoding performance significantly.

## Installation

To run the code in this repository, you need to have Python installed along with the following dependencies:
- PyTorch
- NumPy
- Matplotlib

You can install the required packages using the following command:

```bash
pip install torch numpy matplotlib
```

sage
Clone this repository:

bash
Copy code
git clone https://github.com/YOUR_USERNAME/Neural-Decoding-With-Optimization-of-Node-Activations.git
cd Neural-Decoding-With-Optimization-of-Node-Activations
Run the training script:

bash
Copy code
python train.py
Evaluate the model:

bash
Copy code
python evaluate.py



Neural-Decoding-With-Optimization-of-Node-Activations/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── neural_decoder.py
│   └── ...
│
├── utils/
│   ├── data_loader.py
│   └── ...
│
├── train.py
├── evaluate.py
├── README.md
└── requirements.txt



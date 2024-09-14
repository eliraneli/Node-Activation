# Neural Decoding With Optimization of Node Activations

This repository contains an implementation of the neural decoding technique described in the paper "Neural Decoding With Optimization of Node Activations" by Eliya Nachmani and Yair Be'ery. The paper introduces two novel loss terms to improve the performance of neural decoders for error-correcting codes.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)


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
- PyYAML

You can install the required packages using the following command:

```bash
pip3 install -r requirements.txt
```

## Usage
Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/Neural-Decoding-With-Optimization-of-Node-Activations.git
cd Neural-Decoding-With-Optimization-of-Node-Activations
```
Edit the configuration File and run:
```bash
python main.py
```

## Repository Structure

```
Neural-Decoding-With-Optimization-of-Node-Activations/
│
├── codes/
│   ├── alist files
│   └── npy files
│
├── Decoder/
│   ├── decoder.py
│
├── Code/  
│  ├── code.py
├── utils/
│   ├── config.py
│   └── ...
│
├── config.yaml
├── main.py
├── README.md
└── requirements.txt
```



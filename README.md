# Adversarial Attack Visualization

This repository contains code for training neural networks (Fully Connected, Convolutional, ResNet) and implementing adversarial attacks (L0, L1, L2, Linf). The project also includes methods to visualize the impact of these attacks using K-Nearest Neighbors (KNN) counting and manifold proximity techniques.

## Features

- **Neural Network Training**: Train different types of neural networks including Fully Connected (FC), Convolutional (Conv), and ResNet architectures.
- **Adversarial Attacks**: Implement attacks such as L0, L1, L2, and Linf to evaluate the robustness of trained models.
- **Impact Visualization**: Use KNN counting and manifold proximity methods to visualize how adversarial attacks affect model predictions.

## Requirements

- Python 3.11
- PyTorch
- NumPy
- Matplotlib (for visualization)
- Pytorch Lighting
- art.attacks.evasion art.estimators.classification

## Installation

Clone the repository to your local machine:
`git clone https://github.com/JanullaPolasko/Adversarial-Attack-visualization.git `

Install the required packages:
`pip install -r requirements.txt `

## Usage
To zaisiti aby eveyrythin go smoothly we ensure to put your model name, type model_class ans num_classes to the get_dataset_mapping() in network.py. this function is loading in every script and here it will be lokin for all the nesseceteries. then add neural network class to the network.py or use some of which is already there. If the model is pretreined in torch  - put in get dataset mapping on last position True, otherwise False
To train a model and apply adversarial attacks, run the appropriate scripts. For example, you can start by training a model:
`python train.py`


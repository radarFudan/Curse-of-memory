<div align="center">

# Inverse Approximation Theory for Nonlinear Recurrent Neural Networks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2305.19190-B31B1B.svg)](https://arxiv.org/abs/2305.19190)

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

In this paper, we study the curse of memory phenomenon of RNNs in sequence modelling.

It is shown that, simply adding nonlinear activation such as hardtanh and tanh does not relax the phenomenon.

Using stable parameterisation such as softplus parameterisation can relax the curse of memory and achieve stable approximation.

### Activation does not relax curse of memory

Graph 1(exp + hardtanh RNN)

Graph 2(pol + hardtanh RNN)

### Proper parameterization enables stable approximation

| Parameterisation        | Exp    | Pol      |
| ----------------------- | ------ | -------- |
| Diagonal RNN            | Stable | Unstable |
| Vanilla RNN             | Stable | Unstable |
| Stable Parameterisation | Stable | Stable   |

Notice that here

Graph 3(pol + linear Exp RNN)

Graph 4(pol + linear Softplus RNN)

## Models

Discrete case:
$$h\_{k+1} = h_k + \\Delta t\\sigma(Wh_k+Ux_k)$$

Continuous case:
$$\\frac{dh\_{t}}{dt} = \\sigma(Wh_k+Ux_k)$$

The discrete case can be viewed as an Euler method for the continuous dynamical system.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/radarFudan/Curse-of-memory
cd Curse-of-memory

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/radarFudan/Curse-of-memory
cd Curse-of-memory

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to train

```bash
python src/train.py experiment=Lf/lf-rnn.yaml
```

## Memory function evaluation

```bash
python src/memory.py experiment=Lf/lf-rnn.yaml
```

## Perturbation error evaluation

```bash
python src/perturb.py experiment=Lf/lf-rnn.yaml
```
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

In this paper, we study RNNs' curse of memory phenomenon.

It is shown that, simply adding nonlinear activations such as hardtanh and tanh does not relax the curse.

Using stable parameterisation such as softplus parameterisation can relax the curse of memory and achieve stable approximation for long-memory.

<details>
<summary><b>Curse of memory in linear RNNs</b></summary>

Let $m$ be the hidden dimensions.
We manually construct datasets with different memory patterns.
Short-memory one is exponential decay and long-memory one is polynomial decay ($\rho_t = e^{-t}$ and $\rho_t = \frac{1}{(1+t)^p}$.)

|                                              Exponential decaying memory can be stably approximated                                              |                                              Polynomial decaying memory cannot be stably approximated                                              |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Exponential decaying memory can be stably approximated](https://github.com/radarFudan/Curse-of-memory/blob/main/figs/PerturbationErrorExp.png) | ![Polynomial decaying memory cannot be stably approximated](https://github.com/radarFudan/Curse-of-memory/blob/main/figs/PerturbationErrorPol.png) |

<!-- I don't know why I have to use absolute path here.  -->

</details>

<details>
<summary><b>Curse of memory in nonlinear RNNs</b></summary>

Next, we still work on the polynomial decay memory.
We show that the commonly-used activations (hardtanh and tanh) do not directly relaxed the difficuly in the polynomial decaying memory task.

|                                                         Hardtanh                                                         |                                                       Tanh                                                       |
| :----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| ![Hardtanh does not enable stable approximation](/logs/LF_hardtanh_rnn_pol_PERTURB/runs/20230716/perturbation_error.png) | ![Tanh does not enable stable approximation](/logs/LF_tanh_rnn_pol_PERTURB/runs/20230716/perturbation_error.png) |

</details>

<details>
<summary><b>Proper parameterisation enables stable approximation for long memory</b></summary>

We'll designate the parameterizations that accommodate long-term memory as stable parameterizations.

| Parameterisation        | Exponential decay | Polynomial decay |
| ----------------------- | ----------------- | ---------------- |
| Diagonal RNN            | Stable            | Unstable         |
| Vanilla RNN             | Stable            | Unstable         |
| State space model       | Stable            | Unstable         |
| Linear Recurrent Unit   | Stable            | Unstable         |
| Stable Parameterisation | Stable            | Stable           |
| S4                      | Stable            | Stable           |

|                                                  Vanilla RNN                                                  |                                                      Stable Parameterisation                                                      |
| :-----------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
| ![Vanilla RNN no stable approximation](logs/LF_hardtanh_rnn_pol_PERTURB/runs/20230716/perturbation_error.png) | ![Stable Parameterisation -> stable approximation](logs/LF_hardtanh_softplusrnn_pol_PERTURB/runs/20230716/perturbation_error.png) |

</details>

## Models

### RNNs

Discrete case:
$$h_{k+1} = h_k + \Delta t\sigma(Wh_k+Ux_k).$$

Continuous case:
$$\frac{dh_{t}}{dt} = \sigma(Wh_k+Ux_k).$$

$$y_t = c^\top h_t$$

The discrete case can be viewed as an Euler method for the continuous dynamical system.

### SSMs

The state-space models we are talking about refer to the linear RNNs with layer-wise nonlinear activations. 

$$y_t = f(h_t) \approx W_1 \sigma (W_2 h_t + b_2) + b_1.$$

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

### Perturbation error evaluation

```bash
python src/perturb.py experiment=Lf/lf-rnn.yaml
```

## Future plan

1. Adding state space model, S4, [LRU](https://arxiv.org/abs/2303.06349).
2. Add other RNN variants such as GRU, LSTM, CoRNN, LEM.
3. Add convolutional networks (TCN and Ckconv). 
4. Current sequence length support is around 200. Improve the dataset preparation code so larger sequence length (1000+) can be tested. Need associative scan implementation for the dataset generation. 
5. Docker image creation - delayed. 

## Refs

### Curse of memory / stable approximation / memory functions

```bibtex
@misc{wang2023inverse,
      title={Inverse Approximation Theory for Nonlinear Recurrent Neural Networks},
      author={Shida Wang and Zhong Li and Qianxiao Li},
      year={2023},
      eprint={2305.19190},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Survey on sequence modelling from approximation theory

```bibtex
@Article{JML-2-1,
    author = {Jiang , Haotian Li , Qianxiao Li , Zhong and Wang , Shida},
    title = {A Brief Survey on the Approximation Theory for Sequence Modelling},
    journal = {Journal of Machine Learning},
    year = {2023},
    volume = {2},
    number = {1},
    pages = {1--30},
    abstract = {We survey current developments in the approximation theory of sequence modelling in machine learning. Particular emphasis is placed on classifying existing results for various model architectures through the lens of classical approximation paradigms, and the insights one can gain from these results. We also outline some future research directions towards building a theory of sequence modelling.
    },
    issn = {2790-2048},
    doi = {https://doi.org/10.4208/jml.221221},
    url = {http://global-sci.org/intro/article_detail/jml/21511.html}
}
```

### State-space models

The S4 model was developed by Albert Gu, Karan Goel, and Christopher Ré.
If you find the S4 model useful, please cite their impressive paper:

```bibtex
@misc{gu2021efficiently,
    title={Efficiently Modeling Long Sequences with Structured State Spaces},
    author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
    year={2021},
    eprint={2111.00396},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Also consider checking out their fantastic repository at [github.com/HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces).

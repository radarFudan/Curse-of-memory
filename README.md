<div align="center">

# Inverse Approximation Theory for Nonlinear Recurrent Neural Networks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2305.19190-B31B1B.svg)](https://arxiv.org/abs/2305.19190)
[![Paper](http://img.shields.io/badge/paper-arxiv.2309.13414-B31B1B.svg)](https://arxiv.org/abs/2309.13414)
[![Paper](http://img.shields.io/badge/paper-arxiv.2311.14495-B31B1B.svg)](https://arxiv.org/abs/2311.14495)
[![Conference](http://img.shields.io/badge/NeurIPS-2023-4b44ce.svg)](https://papers.nips.cc/paper/2023)
[![Conference](http://img.shields.io/badge/ICLR-2024-4b44ce.svg)](https://openreview.net/forum?id=yC2waD70Vj)

</div>

## Description

Curse of memory refers to the difficulty of learning long-term memory using recurrent models. 
Although recurrent models benefit from low inference costs, this curse restricts their effectiveness for tasks involving long sequences.
In this paper, we study nonlinear RNNs' curse of memory phenomenon.
It is shown that, simply adding nonlinear activations such as hardtanh and tanh does not relax the curse.
Using stable reparameterisation such as exp parameterisation and softplus parameterisation can relax the curse of memory and achieve stable approximation for long-term memories.

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

| Parameterisation          | Exponential decay | Polynomial decay |
| ------------------------- | ----------------- | ---------------- |
| Diagonal RNN              | Stable            | Unstable         |
| Vanilla RNN               | Stable            | Unstable         |
| State-space model         | Stable            | Unstable         |
| Linear Recurrent Unit     | Stable            | Unstable         |
| Stable Reparameterisation | Stable            | Stable           |

|                                                  Vanilla RNN                                                  |                                                      Stable Parameterisation                                                      |
| :-----------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
| ![Vanilla RNN no stable approximation](/logs/LF_hardtanh_rnn_pol_PERTURB/runs/20230716/perturbation_error.png) | ![Stable Parameterisation -> stable approximation](/logs/LF_hardtanh_softplusrnn_pol_PERTURB/runs/20230716/perturbation_error.png) |

</details>

## Models

### Nonlinear RNNs

Discrete-time case:
$$h_{k+1} = h_k + \Delta t \mathbf{\sigma}(Wh_k+Ux_k+b)$$

$$y_k = c^\top h_k$$

Continuous-time case:
$$\frac{dh_{t}}{dt} = \mathbf{\sigma}(Wh_t+Ux_t+b)$$

$$y_t = c^\top h_t$$

### SSMs

State-space models refer to the linear RNNs with layer-wise nonlinear activations.

Discrete-time case:
$$h_{k+1} = Wh_k+Ux_k+b$$

$$y_k = c^\top \mathbf{\sigma}(h_k)$$

Continuous-time case:
$$\frac{dh_{t}}{dt} = Wh_t+Ux_t+b$$

$$y_t = c^\top \mathbf{\sigma}(h_t)$$

### StableSSM

State-space models refer to the linear RNNs with layer-wise nonlinear activations.

Discrete-time case:
$$h_{k+1} = \Lambda h_k+Ux_k+b$$

$$\Lambda = f(W)$$
$W$ is the trainable weight. $f$ can be exponential/softplus/inverse parameterization.



## Installation

#### Pip

```bash
# clone project
git clone https://github.com/radarFudan/Curse-of-memory
cd Curse-of-memory
conda create -n CoM python=3.10
conda activate CoM
pip install -r requirements.txt
```


## Refs

#### Curse of memory phenomenon / definition of memory function and stable approximation

```bibtex
@inproceedings{
    wang2024inverse,
    title={Inverse Approximation Theory for Nonlinear Recurrent Neural Networks},
    author={Shida Wang and Zhong Li and Qianxiao Li},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=yC2waD70Vj}
}
@inproceedings{
    li2021on,
    title={On the Curse of Memory in Recurrent Neural Networks: Approximation and Optimization Analysis},
    author={Zhong Li and Jiequn Han and Weinan E and Qianxiao Li},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=8Sqhl-nF50}
}
```

#### Extension to state-space models
```bibtex
@inproceedings{
    wang2023statespace,
    title={State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory},
    author={Shida Wang and Beichen Xue},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=i0OmcF14Kf}
}
@misc{wang2023stablessm,
    title={StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization},
    author={Shida Wang and Qianxiao Li},
    year={2023},
    eprint={2311.14495},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### Survey on sequence modelling from approximation perspective

```bibtex
@Article{JML-2-1,
    author = {Haotian Jiang and Qianxiao Li and Zhong Li and Shida Wang},
    title = {A Brief Survey on the Approximation Theory for Sequence Modelling},
    journal = {Journal of Machine Learning},
    year = {2023},
    volume = {2},
    number = {1},
    pages = {1--30},
    abstract = {We survey current developments in the approximation theory of sequence modelling in machine learning. Particular emphasis is placed on classifying existing results for various model architectures through the lens of classical approximation paradigms, and the insights one can gain from these results. We also outline some future research directions towards building a theory of sequence modelling.},
    issn = {2790-2048},
    doi = {https://doi.org/10.4208/jml.221221},
    url = {http://global-sci.org/intro/article_detail/jml/21511.html} }
```

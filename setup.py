#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Curse-of-memory in Nonlinear RNNs",
    author="Wang Shida",
    author_email="e0622338@u.nus.edu",
    url="https://github.com/radarFudan/Curse-of-memory",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
            "perturb_command = src.perturb:main",
        ]
    },
)

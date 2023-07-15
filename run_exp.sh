#!/bin/bash

cd ./wsd/Curse-of-memory

pip install -r requirements.txt --upgrade

wandb offline

bash scripts/lf_exp.sh

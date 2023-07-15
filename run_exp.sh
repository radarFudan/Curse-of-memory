#!/bin/bash

cd ./wsd/Curse-of-memory

pip install -r requirements.txt

wandb offline

bash scripts/lf_exp.sh

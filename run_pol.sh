#!/bin/bash

cd ./wsd/Curse-of-memory || exit

pip install -r requirements.txt

wandb offline

bash scripts/lf_pol.sh

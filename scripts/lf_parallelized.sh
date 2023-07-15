#!/bin/bash

# full
activation_list=("linear" "hardtanh" "tanh") # recurrent activation
model_list=("diagonalrnn" "rnn" "softplusrnn") # parameterization method
rho_name_list=("exp" "pol")

# Debug, two process
# activation_list=("linear")
# model_list=("diagonalrnn" "rnn" "softplusrnn")
# rho_name_list=("exp" "pol")

train_and_perturb() {
    activation=$1
    model=$2
    rho_name=$3
    experiment="LF/lf-${model}"

    task_name="LF_${activation}_${model}_${rho_name}"
    log_dir_path="logs/${task_name}/runs"

    rec1_size_list=("8" "16" "32" "64")
    metric_value=("100")
    ckpt_path_file="${log_dir_path}/ckpt_path.txt"
    trained=False

    gpu_index=$(((PARALLEL_SEQ-1)%4))  # Cycle through GPU indices 0,1,2,3
    export CUDA_VISIBLE_DEVICES=$gpu_index  # Set CUDA_VISIBLE_DEVICES for this job

    if [ "$trained" = False ]
    then
        for i in "${!rec1_size_list[@]}"
        do
            python src/train.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}" callbacks.early_stopping.stopping_threshold="${metric_value[-1]}" logger=many_loggers

            metric_value+=("$(cat "${log_dir_path}/metric_value.txt")")
            echo "Ensure approximation: "
            echo "${metric_value[@]}"
        done
    else
        echo "The model is already trained."
    fi

    # Read checkpoint paths
    ckpt_path=()
    while IFS= read -r line
    do
        ckpt_path+=("$line")
    done < "$ckpt_path_file"

    echo "checkpoints paths:"
    echo "${ckpt_path[@]}"

    for i in "${!ckpt_path[@]}"
    do
        python src/perturb.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}_PERTURB" logger=many_loggers ckpt_path="${ckpt_path[$i]}" +model.net.training=False +perturb_range=21 +perturb_repeats=3 +perturbation_interval=0.5
    done
}

export -f train_and_perturb

# shellcheck disable=SC1083
parallel -j 8 train_and_perturb {1} {2} {3} ::: "${activation_list[@]}" ::: "${model_list[@]}" ::: "${rho_name_list[@]}"

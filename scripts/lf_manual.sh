#!/bin/bash

activation_list=("hardtanh") # recurrent activation
model_list=("diagonalrnn") # parameterization method
rho_name_list=("exp")

for activation in "${activation_list[@]}"
do
    for model in "${model_list[@]}"
    do
        for rho_name in "${rho_name_list[@]}"
        do
            experiment="LF/lf-${model}"

            task_name="LF_${activation}_${model}_${rho_name}"
            log_dir_path="logs/${task_name}/runs"

            rec1_size_list=("8" "16" "32" "64")
            metric_value=("100")
            ckpt_path_file="${log_dir_path}/ckpt_path.txt"
            echo "clean up ckpt_path_file and metric_value"
            echo -n > "$ckpt_path_file"
            echo -n > "${log_dir_path}/metric_value.txt"
            trained=False

            gpu_index=0

            if [ "$trained" = False ]
            then
                for i in "${!rec1_size_list[@]}"
                do
                    CUDA_VISIBLE_DEVICES=$gpu_index python src/train.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}" callbacks.early_stopping.stopping_threshold="${metric_value[-1]}" logger=many_loggers

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
                CUDA_VISIBLE_DEVICES=$gpu_index python src/perturb.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}_PERTURB" logger=many_loggers ckpt_path="${ckpt_path[$i]}" +model.net.training=False +perturb_range=21 +perturb_repeats=3 +perturbation_interval=0.5
            done
        done
    done
done

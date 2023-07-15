#!/bin/bash

activation_list=("linear" "hardtanh" "tanh") # recurrent activation
model_list=("diagonalrnn" "rnn" "softplusrnn") # parameterization method
rho_name_list=("exp" "pol")

for activation in "${activation_list[@]}"
do
    for model in "${model_list[@]}"
    do
        for rho_name in "${rho_name_list[@]}"
        do
            experiment="LF/lf-${model}"
            task_name="LF_${activation}_${model}_${rho_name}"
            log_dir_path="logs/${task_name}/runs/"

            rec1_size_list=("8" "16" "32" "64")
            metric_value=("100")
            ckpt_path=()
            ckpt_path_file="${log_dir_path}/ckpt_path.txt"
            trained=False

            if [ "$trained" = False ]
            then
                for i in "${!rec1_size_list[@]}"
                do
                    python src/train.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}" callbacks.early_stopping.stopping_threshold="${metric_value[-1]}" logger=many_loggers

                    ckpt_path+=("$(cat ckpt_path.txt)")
                    echo "${ckpt_path[-1]}" >> "$ckpt_path_file"

                    metric_value+=("$(cat metric_value.txt)")
                    echo "Ensure approximation: "
                    echo "${metric_value[@]}"
                done
            else
                echo "The model is already trained."
                while IFS= read -r line
                do
                    ckpt_path+=("$line")
                done < "$ckpt_path_file"
            fi


            for i in "${!rec1_size_list[@]}"
            do
                python src/perturb.py experiment="${experiment}" data.rho_name="${rho_name}" model.net.rec1_size="${rec1_size_list[$i]}" model.net.activation="${activation}" task_name="${task_name}_PERTURB" ckpt_path="${ckpt_path[$i]}" +model.net.training=False +perturb_range=21 +perturb_repeats=30 +perturbation_interval=0.5
            done
        done
    done
done

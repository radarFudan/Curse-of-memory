import os
import numpy as np
import matplotlib.pyplot as plt

import torch


def Filter_generate(
    data_dir,
    size,
    seq_length,
    input_dim,
    dt,
    rho,
    target_name,
    Gaussian_input=False,
    evaluate_inputs=None,
):
    r"""Dataset generation for linear functional.

    H_t(x) = \int_{0}^t rho(t-s) x(s) ds
    """

    if data_dir is not None:
        input_file_path = data_dir + f"filter_{target_name}_inputs.npy"
        output_file_path = data_dir + f"filter_{target_name}_outputs.npy"
        memory_file_path = data_dir + f"filter_{target_name}_memory.npy"

        # Skip generation if files exist
        if os.path.isfile(input_file_path) and os.path.isfile(output_file_path):
            print("Files for filter already exist, skipping generation.")
            inputs = np.load(input_file_path)
            output_reshaped = np.load(output_file_path)
        else:
            inputs = dt * np.random.normal(size=(size, seq_length, input_dim))

            # Make input Gaussian process
            if Gaussian_input:
                inputs = np.cumsum(inputs, axis=1)

            # TODO
            # Construct rnn or attention with linear bias based on the target
            hidden_size=16

            if target_name == "rnn":
                model = torch.nn.RNN(input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, dtype=torch.float64)
            elif target_name == "transformer":
                model = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=hidden_size, dim_feedforward=hidden_size*2, nhead=1, dtype=torch.float64), num_layers=1)
            else:
                raise NotImplementedError

            readin = torch.nn.Linear(input_dim, hidden_size, dtype=torch.float64)
            readout = torch.nn.Linear(hidden_size, input_dim, dtype=torch.float64)


            query_inputs = np.zeros((1, seq_length, input_dim))
            
            if target_name == "rnn":
                output_reshaped, _ = model(torch.tensor(inputs))
                query_outputs, _ = model(torch.tensor(query_inputs))
            elif target_name == "transformer":
                output_reshaped = model(readin(torch.tensor(inputs)), readin(torch.tensor(inputs)))
                query_outputs = model(readin(torch.tensor(query_inputs)), readin(torch.tensor(query_inputs)))
            else:
                raise NotImplementedError

            output_reshaped = readout(output_reshaped)

            np.save(input_file_path, inputs)
            np.save(output_file_path, output_reshaped.detach().numpy())

            query_inputs[0, 0, :] = 1
            query_outputs = readout(query_outputs)
            np.save(memory_file_path, query_outputs.detach().numpy())

            query_outputs_squeezed = np.squeeze(query_outputs.detach().numpy())

            query_outputs_squeezed_memory = np.abs(query_outputs_squeezed[1:] - query_outputs_squeezed[:-1])

            plt.plot(query_outputs_squeezed_memory)
            plt.yscale("log")
            plt.savefig(data_dir + f"filter_{target_name}_memory.png")
            plt.savefig(data_dir + f"filter_{target_name}_memory.pdf")
            plt.close()


        # Normalize
        # inputs /= np.max(np.abs(inputs))
        # outputs /= np.max(np.abs(outputs))

        # print("filter_generate done")
        # print("In filter datagen, input shape", inputs.shape)
        # print("In filter datagen, output shape", output_reshaped.shape)

    # if evaluate_inputs is not None:
    #     # assert evaluate_inputs.shape[1] == seq_length  # Sequence length not necessarily the same
    #     assert evaluate_inputs.shape[2] == input_dim
    #     seq_length = evaluate_inputs.shape[1]

    #     outputs = []
    #     for t in range(seq_length):
    #         output = 0
    #         for s in range(t + 1):
    #             output += evaluate_inputs[:, t - s, :] * (rho(s * dt))
    #         outputs.append(output)

    #     # print(outputs)
    #     # print(outputs[0].shape)
    #     # print(np.asarray(outputs).shape)
    #     # outputs is of shape (seq_length, size, output_dim), need transpose
    #     output_reshaped = np.asarray(outputs).transpose(1, 0, 2)

    #     return output_reshaped


# Code refactor with filter_datamodule into some new memory register
targets = {
    "rnn": None,
    "transformer": None,
}

if __name__ == "__main__":
    # TODO

    evaluate_inputs = np.ones((1, 10, 1))
    # evaluate_outputs = Filter_generate(
    #     None,
    #     None,
    #     None,
    #     1,
    #     1.0,
    #     targets["rnn"],
    #     None,
    #     Gaussian_input=False,
    #     evaluate_inputs=evaluate_inputs,
    # )
    # print(evaluate_inputs.dtype, evaluate_outputs.dtype)
    evaluate_inputs = np.squeeze(evaluate_inputs)
    # evaluate_outputs = np.squeeze(evaluate_outputs)

    # print("memory", np.abs(evaluate_outputs[1:] - evaluate_outputs[:-1]))

    print("Test passed.")

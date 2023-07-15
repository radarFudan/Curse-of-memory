import os

import numpy as np
import scipy


def LF_generate(
    data_dir,
    size,
    seq_length,
    input_dim,
    dt,
    rho,
    rho_name,
    Gaussian_input=False,
    evaluate_inputs=None,
):
    r"""Dataset generation for linear functional.

    H_t(x) = \int_{0}^t rho(t-s) x(s) ds
    """

    if data_dir is not None:
        input_file_path = data_dir + f"lf_{rho_name}_inputs.npy"
        output_file_path = data_dir + f"lf_{rho_name}_outputs.npy"

        # Skip generation if files exist
        if os.path.isfile(input_file_path) and os.path.isfile(output_file_path) and False:
            print("Files for lf already exist, skipping generation.")
            inputs = np.load(input_file_path)
            outputs = np.load(output_file_path)
        else:
            inputs = dt * np.random.normal(size=(size, seq_length, input_dim))

            # Make input Gaussian process
            if Gaussian_input:
                inputs = np.cumsum(inputs, axis=1)

            outputs = []
            for t in range(seq_length):
                output = 0
                for s in range(t + 1):
                    output += inputs[:, t - s, :] * (rho(s * dt))
                outputs.append(output)

            # outputs is of shape (seq_length, size, output_dim), need transpose
            output_reshaped = np.asarray(outputs).transpose(1, 0, 2)

            np.save(data_dir + f"lf_{rho_name}_inputs.npy", inputs)
            np.save(data_dir + f"lf_{rho_name}_outputs.npy", output_reshaped)

        # Normalize
        # inputs /= np.max(np.abs(inputs))
        # outputs /= np.max(np.abs(outputs))

        # print("LF_generate done")
        # print("In lf datagen, input shape", inputs.shape)
        # print("In lf datagen, output shape", output_reshaped.shape)

    if evaluate_inputs is not None:
        # assert evaluate_inputs.shape[1] == seq_length  # Sequence length not necessarily the same
        assert evaluate_inputs.shape[2] == input_dim
        seq_length = evaluate_inputs.shape[1]

        outputs = []
        for t in range(seq_length):
            output = 0
            for s in range(t + 1):
                output += evaluate_inputs[:, t - s, :] * (rho(s * dt))
            outputs.append(output)

        # print(outputs)
        # print(outputs[0].shape)
        # print(np.asarray(outputs).shape)
        # outputs is of shape (seq_length, size, output_dim), need transpose
        output_reshaped = np.asarray(outputs).transpose(1, 0, 2)

        return output_reshaped


# Code refactor with lf_datamodule into some new memory register
rhos = {
    "exp": lambda t: np.exp(-t),
    "pol": lambda t: 1 / (1 + 0.1 * t) ** 1.1,
    "shift": lambda t: 1 if (t > 9) and (t < 11) else 0,
    "twoparts": lambda t: np.exp(-t) + np.exp(-((t - 9.0) ** 2)),
    "airy": lambda t: scipy.special.airy(t - 6.0)[0],
    "sin": lambda t: np.sin(t),
}

if __name__ == "__main__":
    # TODO

    evaluate_inputs = np.ones((1, 10, 1))
    evaluate_outputs = LF_generate(
        None,
        None,
        None,
        1,
        1.0,
        rhos["exp"],
        None,
        Gaussian_input=False,
        evaluate_inputs=evaluate_inputs,
    )
    print(evaluate_inputs.dtype, evaluate_outputs.dtype)
    evaluate_inputs = np.squeeze(evaluate_inputs)
    evaluate_outputs = np.squeeze(evaluate_outputs)

    # print("memory", np.abs(evaluate_outputs[1:] - evaluate_outputs[:-1]))

    print("Test passed.")

import torch
import numpy as np
from model_architecture import NeuralNetwork

# --- Parameters Settings ---
node_numbers = (9, 9)

force = np.zeros((2, node_numbers[0], node_numbers[1]))
force[0, -1, -1] = 100
force = force.reshape(
    (-1),
    order="F",
)

volume_fraction = 0.5

fix_boundary = np.array(
    [1] * 2 * node_numbers[1] + [0] * 2 *
    (node_numbers[0] - 1) * node_numbers[1],
    dtype=np.float32,
)

# --- Parameters Settings ---

if __name__ == "__main__":
    input_x = torch.tensor(
        np.array([
            np.concatenate(
                (
                    [fix_boundary],
                    [force],
                    [[volume_fraction]],
                ),
                axis=1,
            )
        ]),
        dtype=torch.float32,
    ).reshape(1, -1)

    model = NeuralNetwork()
    model.load_state_dict(torch.load("src/model_saved/model_8_8_2022-07-18 15:00:05.126287.pth"))
    model.eval()

    optimized_density = model(input_x)[0]

    print(optimized_density)
    print(torch.mean(optimized_density))

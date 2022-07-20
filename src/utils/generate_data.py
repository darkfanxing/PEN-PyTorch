import numpy as np


def generate_data(basic_design_domain, data_size, level):
    basic_node_numbers = [basic_design_domain[0] + 1, basic_design_domain[1] + 1]
    level_node_numbers = [
        basic_design_domain[0] * 2**level + 1, basic_design_domain[1] * 2**level + 1
    ]

    volume_fractions = np.around(
        np.random.uniform(0.2, 0.81, size=(data_size)), 2)

    radians = np.deg2rad(
        np.around(np.random.uniform(
            0,
            360,
            size=(data_size),
        )))

    force_values = np.array([100 * np.cos(radians), 100 * np.sin(radians)])

    force_x_indices = np.random.randint(
        low=1,
        high=basic_node_numbers[0],
        size=(data_size),
    )

    force_y_indices = np.random.randint(
        low=0,
        high=basic_node_numbers[1],
        size=(data_size),
    )

    basic_force_matrices = np.zeros(
        (data_size, 2, basic_node_numbers[0], basic_node_numbers[1]))
    level_force_matrices = np.zeros(
        (data_size, 2, level_node_numbers[0], level_node_numbers[1]))

    for index in range(data_size):
        basic_force_matrices[index, :, force_x_indices[index], force_y_indices[index]] = \
            force_values[:, index]

        level_force_matrices[index, :, force_x_indices[index] * 2**level, force_y_indices[index] * 2**level] = \
            force_values[:, index]

    return {
        "volume_fractions": volume_fractions,
        "force": {
            "vectors": {
                "basic": basic_force_matrices.reshape(
                    (data_size, -1),
                    order="F",
                ),
                "level": level_force_matrices.reshape(
                    (data_size, -1),
                    order="F",
                ),
            }
        },
        "fixed_boundary": {
            "level_indices":
            np.arange(2 * level_node_numbers[1]),
            "basic_vector":
            np.array([1] * 2 * basic_node_numbers[1] + [0] * 2 *
                     (basic_node_numbers[0] - 1) * basic_node_numbers[1]),
        }
    }

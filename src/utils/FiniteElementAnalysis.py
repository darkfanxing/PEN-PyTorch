from numpy import ndarray, setdiff1d, arange, array, zeros, ones, kron, dot, newaxis, float32
from typing import Tuple
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class FiniteElementAnalysis():

    def __init__(
        self,
        youngs_modulus: float,
        poisson_ratio: float,
        design_domain: Tuple[int, int],
        fixed_indices: ndarray,
        force_vector: ndarray,
        penalization: float = 3,
        min_density: float = 1e-3,
    ) -> None:
        self.design_domain = design_domain
        self.penalization = penalization
        self.youngs_modulus = youngs_modulus
        self.fixed_indices = fixed_indices
        self.force_vector = force_vector
        self.min_density = min_density

        self.d_matrix = self.get_d_matrix(youngs_modulus, poisson_ratio)

    def solve(self, density):
        dof_number = \
            2 * (self.design_domain[0] + 1) * (self.design_domain[1] + 1)

        dof_free_indices = setdiff1d(
            arange(dof_number),
            self.fixed_indices,
        )

        elements_dof = zeros(
            (
                self.design_domain[0] * self.design_domain[1],
                8,
            ),
            dtype=int,
        )

        for number_x in range(self.design_domain[0]):
            for number_y in range(self.design_domain[1]):
                element_number = number_y + number_x * self.design_domain[1]
                node_1_number = \
                    (self.design_domain[1] + 1) * number_x + number_y
                node_2_number = \
                    (self.design_domain[1] + 1) * (number_x + 1) + number_y

                elements_dof[element_number, :] = array([
                    2 * node_1_number + 2, 2 * node_1_number + 3,
                    2 * node_2_number + 2, 2 * node_2_number + 3,
                    2 * node_2_number, 2 * node_2_number + 1,
                    2 * node_1_number, 2 * node_1_number + 1
                ])

        element_position_x = kron(elements_dof, ones((8, 1))).flatten()
        element_position_y = kron(elements_dof, ones((1, 8))).flatten()
        stiffness_values = (self.d_matrix.flatten()[newaxis].T *
                            (1e-9 +
                             (0.01 + density.flatten())**self.penalization *
                             (self.youngs_modulus - 1e-9))).flatten(order="F")

        stiffness_matrix = coo_matrix(
            (stiffness_values, (element_position_x, element_position_y)),
            shape=(dof_number, dof_number),
            dtype=float32).tocsc()

        stiffness_matrix = \
            stiffness_matrix[dof_free_indices, :][:, dof_free_indices]

        displacement_vector = zeros((dof_number, 1))
        displacement_vector[dof_free_indices, 0] = spsolve(
            stiffness_matrix,
            self.force_vector[dof_free_indices],
        )

        elements_objective_function_value = (dot(
            displacement_vector[elements_dof].reshape(
                (self.design_domain[0] * self.design_domain[1], 8)),
            self.d_matrix,
        ) * displacement_vector[elements_dof].reshape(
            (self.design_domain[0] * self.design_domain[1], 8))).sum(axis=1)

        return elements_objective_function_value

    def get_d_matrix(self, youngs_modulus, poisson_ratio):
        # element stiffness
        k = [
            (0.5 - poisson_ratio / 6),
            (0.125 + poisson_ratio / 8),
            (-0.25 - poisson_ratio / 12),
            (-0.125 + 3 * poisson_ratio / 8),
            (-0.25 + poisson_ratio / 12),
            (-0.125 - poisson_ratio / 8),
            (poisson_ratio / 6),
            (0.125 - 3 * poisson_ratio / 8),
        ]

        return youngs_modulus / (1 - poisson_ratio**2) * array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ])

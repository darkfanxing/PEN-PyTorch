from utils import FiniteElementAnalysis, Loss, generate_data
from model_architecture.NeuralNetwork import NeuralNetwork

from typing import List

import numpy as np

import torch
import torch.optim as optim

import time


class PEN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_sizes = [2048, 512, 128, 8]
    # data_sizes = [2048]
    loss_function = Loss()

    def __init__(
        self,
        youngs_modulus: float,
        poisson_ratio: float,
        basic_design_domain: List[int],
        model_save_path: str,
        penalization=3,
    ):
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.penalization = penalization
        self.basic_design_domain = basic_design_domain
        self.model_save_path = model_save_path

    def fit(self, learning_rate=0.01):
        self.objective_function_values = []

        model = NeuralNetwork().to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        start_time = time.time()

        batch_number = 0
        level_batch_number = [0, 0, 0, 0]
        self.losses = [[], [], [], []]
        for index, data_size in enumerate(self.data_sizes):
            design_domain = list(
                map(lambda x: x * 2**index, self.basic_design_domain))

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.25**index * 0.005

            patience = 0
            best_loss = None
            while (patience < 500):
                batch_number += 1
                level_batch_number[index] += 1

                data = generate_data(
                    self.basic_design_domain,
                    data_size,
                    index + 1,
                )

                input_x = torch.tensor(
                    np.array([
                        np.concatenate(
                            (
                                np.kron(
                                    data["fixed_boundary"]["basic_vector"],
                                    np.ones((data_size, 1)),
                                ),
                                data["force"]["vectors"]["basic"],
                                np.array([data["volume_fractions"]]).reshape(
                                    data_size, 1),
                            ),
                            axis=1,
                        )
                    ]),
                    dtype=torch.float32,
                ).reshape((data_size, -1)).to(self.device)

                self.optimizer.zero_grad()
                densities = model(input_x)[index]

                compliances = []
                for data_index in range(data_size):
                    finite_element_analysis = FiniteElementAnalysis(
                        youngs_modulus=self.youngs_modulus,
                        poisson_ratio=self.poisson_ratio,
                        design_domain=design_domain,
                        fixed_indices=data["fixed_boundary"]["level_indices"],
                        force_vector=data["force"]["vectors"]["level"][
                            data_index, :],
                        penalization=self.penalization,
                    )
                    compliances.append(
                        finite_element_analysis.solve(
                            densities[data_index].cpu().detach().numpy()))

                loss = self.loss_function(
                    densities,
                    torch.tensor(np.array(
                        compliances,
                        dtype=np.float32,
                    ), ).to(self.device),
                    torch.tensor(
                        [data["volume_fractions"]],
                        dtype=torch.float32,
                    ).reshape(data_size, 1).to(self.device),
                    self.penalization,
                    self.device,
                )

                self.losses[index].append(float(loss.detach().cpu().numpy()))

                if patience == 0:
                    best_loss = loss
                    patience += 1
                else:
                    if best_loss - loss > 0.01:
                        best_loss = loss
                        patience = 0
                    else:
                        patience += 1

                loss.backward()
                self.optimizer.step()

                print(batch_number)
                print(level_batch_number)

        print(f"time: {time.time() - start_time}")
        torch.save(model.state_dict(), self.model_save_path)

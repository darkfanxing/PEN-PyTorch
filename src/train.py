from PEN import PEN
import datetime
import matplotlib.pyplot as plt

pen = PEN(
    youngs_modulus=1.95e5,
    poisson_ratio=0.3,
    basic_design_domain=[8, 8],
    model_save_path=f"src/model_saved/model_8_8_{datetime.datetime.now()}.pth",
)

pen.fit()

for index in range(len(pen.losses)):
    plt.plot(list(range(len(pen.losses[index]))), pen.losses[index], label=f"level {index+1}")
plt.legend()
plt.show()

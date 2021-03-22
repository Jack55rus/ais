import matplotlib.pyplot as plt
from immune_system import NegativeSelection
import numpy as np

if __name__ == '__main__':
    # ags = np.array([[.2, .54], [.76, .29]])
    ags_list = []
    for i in range(150):
        ags_list.append([np.random.random(), np.random.random()])
    ags = np.array(ags_list)
    ags_test = np.array([[.65, .45], [.16, .98]])
    nsa = NegativeSelection(num_detectors=250)
    nsa.fit(ags)
    preds = nsa.predict(ags_test)
    circles = []
    for d in nsa.memory:
        # plt.scatter(d.get_params()[1][0], d.get_params()[1][1])
        circles.append(plt.Circle((d.get_params()[1][0], d.get_params()[1][1]),
                             radius=d.get_params()[0], color='r', clip_on=False, fill=False))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ags[:, 0], ags[:, 1])
    # ax.scatter(ags[1][0], ags[1][1])
    for circle in circles:
        ax.add_patch(circle)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 1.2)
    plt.show()


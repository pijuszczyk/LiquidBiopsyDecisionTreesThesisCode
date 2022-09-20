from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import data._KEGG as k


def plot_stds_heatmap():
    X = k.load_matrices(Path(__file__).parent)
    stds = np.std(X, 0)
    const_f = np.sum(stds < 0.1**12)
    var_f = np.sum(stds >= 0.1**12)
    print(const_f)
    print(var_f)
    print(const_f / (const_f+var_f))
    sns.heatmap(stds)
    plt.show()


def plot_stds_hist():
    X = k.load_matrices(Path(__file__).parent)
    stds = np.std(X, 0)
    plt.hist(stds.flatten(), 30)
    plt.ylim(0, 10000)
    plt.xlabel("Odchylenie standardowe")
    plt.ylabel("Liczba cech")
    plt.show()


def get_levels_num():
    from sklearn.preprocessing import OrdinalEncoder
    possibly_cont_vars_num = 0
    X = k.load_matrices(Path(__file__).parent)
    for y in range(X.shape[1]):
        for x in range(X.shape[2]):
            enc = OrdinalEncoder()
            max_level = enc.fit_transform(X[:, y, x].reshape(-1, 1)).max()
            if max_level != 0:
                print(f'[{y}, {x}] {max_level}')
                if max_level > 50:
                    possibly_cont_vars_num += 1
    print(possibly_cont_vars_num)


if __name__ == "__main__":
    # plt.bar(['Cechy stałe', 'Cechy zmienne'], [118707, 23070])
    # plt.show()
    # plot_stds_heatmap()
    plot_stds_hist()
    # get_levels_num()

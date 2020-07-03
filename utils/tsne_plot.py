import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


def TSNE_plot(feat_list, label, save_path):

    color_list = {0: 'gray', 1: 'darkgoldenrod', 2: 'orangered', 3: 'chocolate', 4: 'dodgerblue',
                  5: 'magenta', 6: 'blue', 7: 'saddlebrown', 8: 'chartreuse', 9: 'forestgreen',
                  10: 'aquamarine', 11: 'gold', 12: 'cyan', 13: 'red', 14: 'olive',
                  15: 'teal', 16: 'navy', 17: 'indigo', 18: 'crimson', 250: 'black',
                  19: 'gray', 20: 'darkgoldenrod', 21: 'orangered', 22: 'chocolate', 23: 'dodgerblue',
                  24: 'magenta', 25: 'blue', 26: 'saddlebrown', 27: 'chartreuse', 28: 'forestgreen',
                  29: 'aquamarine', 30: 'gold', 31: 'cyan', 32: 'red', 33: 'olive',
                  34: 'teal', 35: 'navy', 36: 'indigo', 37: 'crimson', 269: 'black'}

    wanted_id = 250
    feat_np = np.array(feat_list)
    label = np.array(label)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50, verbose=1, n_iter=1500)
    X_tsne = tsne.fit_transform(feat_np)

    for x, y in zip(X_tsne, label):
        if y == wanted_id:
            plt.scatter(x[0], x[1], alpha=1.0, color=color_list[y], marker='x')
        else:
            if y not in color_list.keys():
                continue
            plt.scatter(x[0], x[1], alpha=0.2, color=color_list[y], marker='o')
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()
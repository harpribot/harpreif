import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sea
sea.set_style('darkgrid')
sea.set_palette('muted')
sea.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


class Plotter(object):
    def __init__(self, sample_feat, sample_label):
        """
        Plots T-SNE
        :param sample_feat:
        :param sample_label:
        """
        self.sample_feat = np.array(sample_feat)
        self.sample_label = sample_label

    def reduce(self, layer = (100, 2)):
        """

        :param layer:
        :return:
        """
        self.__reduce_pca(layer[0])
        self.__reduce_sne(layer[0], layer[1])

    def __reduce_pca(self, reduced_dim):
        """

        :param reduced_dim:
        :return:
        """
        print "Reducing using PCA..."
        pca = PCA(n_components=reduced_dim, copy=False)
        self.sample_feat = pca.fit_transform(self.sample_feat)
        print self.sample_feat.shape
        assert self.sample_feat.shape[1] == reduced_dim, 'Dimension mismatch after reduction'

    def __reduce_sne(self, input_dim, reduced_dim):
        """

        :param input_dim:
        :param reduced_dim:
        :return:
        """
        print "Reducing using T-SNE..."
        assert self.sample_feat.shape[1] == input_dim, 'Dimension mismatch before reduction'
        tsne = TSNE(n_components=reduced_dim, random_state=0)
        self.sample_feat = tsne.fit_transform(self.sample_feat)
        print self.sample_feat.shape
        assert self.sample_feat.shape[1] == reduced_dim, 'Dimension mismatch after reduction'

    def plot(self):
        """
        Plots a scatter image of the representation of the test data
        :return: None
        """
        print "Plotting..."
        f, ax, sc, txts = Plotter.scatter(self.sample_feat, self.sample_label)
        f.savefig('./images/digits_tsne-generated.png', dpi=120)
        print "Plot saved. All done."

    @staticmethod
    def scatter(x, colors):
        """
        Obtains a scatter plot
        :param x:
        :param colors:
        :return:
        """
        # We choose a color palette with seaborn.
        palette = np.array(sea.color_palette("hls", 258))

        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                patheffects.Stroke(linewidth=5, foreground="w"),
                patheffects.Normal()])
            txts.append(txt)

        plt.show()
        return f, ax, sc, txts

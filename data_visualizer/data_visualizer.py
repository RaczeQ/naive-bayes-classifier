import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data_model import DataModel

FIG_SIZE = (3,2)

def visualize(dataset):
    dm = DataModel.generate_from_file(dataset)
    visualize_histograms(dm, 'histograms-{}'.format(str(dataset)))

def visualize_histograms(data_model, file_name):
    splitted_data = data_model.get_splitted_dataframe()

    features = data_model._continuous_features
    classes = data_model.get_classes_list()
    len_f = len(features)
    len_c = len(classes)

    pp = PdfPages('data_visualizer/visualizations/{}.pdf'.format(file_name))
    fig = plt.figure(figsize=(FIG_SIZE[0] * len_f, FIG_SIZE[1] * (len_c+1)),dpi=200)

    idx = 1
    color=plt.cm.rainbow(np.linspace(0,1,len_c + 1))
    for feature in features:
        ax_f = fig.add_subplot(len_c + 1, len_f, idx)
        if idx == 1:
            ax_f.set_ylabel('All classes')
        ax_f.set_title(feature)
        _, bins, _ = ax_f.hist(data_model._df[feature], density=1, color=color[0])
        for i, c in enumerate(classes):
            ax_c = fig.add_subplot(len_c + 1, len_f, idx + (i+1) * len_f, sharex = ax_f)
            if idx == 1:
                ax_c.set_ylabel(c)
            ax_c.hist(splitted_data[c][feature], density=1, bins=bins, color=color[i+1])
        idx += 1

    fig.tight_layout()
    pp.savefig()
    pp.close()

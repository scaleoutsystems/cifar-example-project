import os
import matplotlib.pyplot as plt
import itertools
import numpy
import sys


def plot_classification_report(classification_report, title='Classification report', c_map='RdBu'):
    classification_report = classification_report.replace('\n\n', '\n')
    classification_report = classification_report.replace(' / ', '/')
    lines = classification_report.split('\n')

    classes, plot_mat, support, class_names = [], [], [], []
    for line in lines[1:]:  # includes avg/total result; otherwise, change [1:] to [1:-1]
        t = line.strip().replace(' avg', '-avg').split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plot_mat.append(v)

    # TODO(desislava@scaleoutsystems.com): Fix a better visualisation for the missing values (0.0 might be confusing).
    max_len = len(max(plot_mat, key=len))
    for x in plot_mat:
        if len(x) == max_len:
            pass
        else:
            while len(x) < max_len:
                x.insert(0, 0.0)

    plot_mat = numpy.array([numpy.array(x) for x in plot_mat])

    x_labels = ['Precision', 'Recall', 'F1-score']
    y_labels = ['{0} ({1})'.format(class_names[idx], sup)
                for idx, sup in enumerate(support)]

    plt.imshow(plot_mat, interpolation='nearest', cmap=c_map, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(numpy.arange(3), x_labels, rotation=45)
    plt.yticks(numpy.arange(len(classes)), y_labels)

    upper_thresh = plot_mat.min() + (plot_mat.max() - plot_mat.min()) / 10 * 8
    lower_thresh = plot_mat.min() + (plot_mat.max() - plot_mat.min()) / 10 * 2
    for i, j in itertools.product(range(plot_mat.shape[0]), range(plot_mat.shape[1])):
        plt.text(j, i, format(plot_mat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plot_mat[i, j] > upper_thresh or plot_mat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()


if __name__ == '__main__':
    classification_report = str(sys.argv[1])
    report_id = int(sys.argv[2])

    plot_classification_report(classification_report)

    plot_name = 'report_{}.png'.format(report_id)

    if not os.path.exists('static/reports/{}'.format(plot_name)):
        plt.savefig('static/reports/{}'.format(plot_name))

    plt.close()

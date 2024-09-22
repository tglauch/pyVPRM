# /usr/bin/python


import matplotlib as mpl
import numpy as np
import scipy as sp

mpl.use("pdf")


def figsize(scale, ratio=(np.sqrt(5.0) - 1.0) / 2.0):
    fig_width_pt = 455.8843  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    #    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    #    "text.usetex": True,                # use LaTeX to write all text
    #    "font.family": "serif",
    #    "font.serif": "Computer Modern",                   # blank entries should cause plots to inherit fonts from the document
    #    "font.sans-serif": [],
    #    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "xtick.major.pad": 10.0,
    "ytick.major.pad": 10.0,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.0,
    "figure.autolayout": True,
    #    "pgf.preamble": [
    #        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
    #        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    #        r"\usepackage[detect-all]{siunitx}"
    #        ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


def newfig(width, ratio=(np.sqrt(5.0) - 1.0) / 2.0, **pargs):
    fig = plt.figure(figsize=figsize(width, ratio))
    plt.clf()
    ax = fig.add_subplot(111, **pargs)
    return fig, ax


colors = [
    "#0065BD",
    "#005293",
    "#003359",
    "#DAD7CB",
    "#E37222",
    "#A2AD00",
    "#98C6EA",
    "#64A0C8",
    "#CCCCC6",
    "#808080",
    "#000000",
]


def setNewEdges(edges):
    newEdges = []
    for i in range(0, len(edges) - 1):
        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
        newEdges.append(newVal)
    return np.array(newEdges)

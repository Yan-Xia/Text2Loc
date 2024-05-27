import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(metrics, save_path, title="", show_plot=False, size=(16, 10)):
    rows = int(np.round(np.sqrt(len(metrics))))
    cols = int(np.ceil(len(metrics) / rows))

    fig = plt.figure()
    fig.set_size_inches(*size)
    plt.title(title)

    for i, key in enumerate(metrics.keys()):
        plt.subplot(rows, cols, i + 1)
        for k in metrics[key].keys():
            l = metrics[key][k]
            (line,) = plt.plot(l)
            if type(k) in (list, tuple):
                line.set_label(k)
            else:
                line.set_label(f"{k:0.6}")
        plt.title(key)
        plt.gca().set_ylim(bottom=0.0)  # Set the bottom to 0.0
        plt.legend()

    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f"\n Plot saved as {save_path} \n")


if __name__ == "__main__":
    metrics = {}
    metrics["metric a"] = {1.0: np.arange(5), 2.0: np.arange(5) + 2}
    metrics["metric b"] = {1.0: np.arange(5), 2.0: np.arange(5) + 2}
    metrics["metric c"] = {1.0: np.arange(5), 2.0: np.arange(5) + 2}

    plot_metrics(metrics, "testfig.png")

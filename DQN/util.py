from matplotlib import pyplot as plt
import numpy as np




def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

    color = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Epsilon', color=color)
    ax1.plot(x, epsilons, color=color, label='Epsilon')
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding a second y-axis to plot the running average score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Score', color=color)

    # Calculating the running average of scores
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100):(t + 1)])

    ax2.scatter(x, running_avg, color=color, label='Score (Running Avg)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding a legend to the plot
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Adding vertical lines if specified
    if lines is not None:
        for line in lines:
            plt.axvline(x=line, color='grey', linestyle='--')

    fig.tight_layout()  # Adjust the layout to make room for the second y-axis
    plt.savefig(filename)
    plt.show()

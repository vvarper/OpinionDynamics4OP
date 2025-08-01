import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


def main():
    concern_threshold = 0.7
    filename = 'data/mig/mig01.csv'

    generator = np.random.default_rng(0)
    perception = pd.read_csv(filename)[['perception']]

    mean_0 = concern_threshold / 2
    sigma_0 = mean_0 / 3

    top_range = (1 - concern_threshold) / 3
    mean_3 = concern_threshold + top_range / 2
    mean_2 = mean_3 + top_range
    mean_1 = mean_2 + top_range
    sigma_top = top_range / 6

    # Define the means and sigmas for each perception
    means = [mean_0, mean_1, mean_2, mean_3]
    sigmas = [sigma_0, sigma_top, sigma_top, sigma_top]
    clips = [(0, concern_threshold), (concern_threshold + 2 * top_range, 1),
             (concern_threshold + top_range,
              concern_threshold + 2 * top_range),
             (concern_threshold, concern_threshold + top_range)]

    # Initialize a list to store the opinions
    opinions = []
    legends = ['Not mentioned', '1st problem', '2nd problem', '3rd problem']

    # Plot the normal distribution (density function) corresponding to each mean and sigma
    for i in range(4):
        x = np.linspace(means[i] - 3 * sigmas[i], means[i] + 3 * sigmas[i],
                        100)
        # Plot coloring the area under the curve

        distribution = norm.pdf(x, means[i], sigmas[i]) * sigmas[i]
        # Scale the distribution to have a maximum of 1
        distribution = distribution / max(distribution)

        plt.fill_between(x, distribution, alpha=0.5, label=legends[i])

        # Repeat previous plot but remarking borders
        plt.plot(x, distribution)

    plt.xlabel('Initial opinions')
    plt.ylabel('Density')
    # Place the legend in the upper left corner
    plt.legend(loc='upper left')

    plt.xticks([0, 0.7 / 2, 0.7, 1],
               [0, r'$\frac{C_{th}}{2}$', r'$C_{th}$', 1])

    plt.tight_layout()
    plt.savefig('results/initial_opinions.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()

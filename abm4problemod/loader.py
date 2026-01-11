import numpy as np
import pandas as pd


def load_opinions(filename, concern_threshold):
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
    clips = [(0, concern_threshold), (concern_threshold + 2 * top_range, 1), (
        concern_threshold + top_range, concern_threshold + 2 * top_range),
             (concern_threshold, concern_threshold + top_range)]

    # Initialize a list to store the opinions
    opinions = []

    # Loop over each perception
    for i in range(4):
        # Get the number of perceptions equal to i
        num = int(perception['perception'].value_counts()[i])

        # Generate random opinions for this perception
        op = generator.normal(means[i], sigmas[i], num).clip(*clips[i])

        # Add the opinions to the list
        opinions.extend(op)

    return opinions


def load_history(filename, variable='concerned'):
    return pd.read_csv(filename)[[variable]].to_numpy().flatten()


def get_period_change_steps(daily_steps, folder='mig'):
    days_per_period = pd.read_csv(f'data/{folder}/simulation_periods.csv')['Days']
    days_per_period = days_per_period[days_per_period != 0]

    steps_per_month = [i * daily_steps for i in days_per_period]

    measure_time = [sum(steps_per_month[:i]) for i in
                    range(0, len(steps_per_month) + 1)]

    return measure_time

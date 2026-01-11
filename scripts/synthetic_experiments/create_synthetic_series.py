import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    offset = 0.25
    time_steps = 16
    t = np.arange(time_steps)
    rng = np.random.default_rng(seed=42)

    # 1. Changes by stable periods
    series = np.array([offset] * 3 + [0.40] * 4 + [0.15] * 5 + [0.35] * 4)
    plt.plot(series)
    plt.title('Synthetic Time Series (Stable Periods)')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid()
    plt.show()
    plt.clf()

    # Save series as a csv
    df = pd.DataFrame(series, columns=['concerned'])
    df.to_csv('data/synth/stable_history.csv', index_label='Month')

    # 2. Sinusoidal changes (1)

    amplitude = 0.15
    f = 1 / 8
    phi = 0
    series = offset + amplitude * np.sin(2 * np.pi * f * t + phi)
    plt.plot(series)
    plt.title('Sinusoidal Synthetic Time Series (1)')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid()
    plt.show()
    plt.clf()

    df = pd.DataFrame(series, columns=['concerned'])
    df.to_csv('data/synth/sin1_history.csv', index_label='Month')

    # 3. Sinuosoidal changes (1) with noise
    sigma_obs = 0.1
    noise = rng.normal(0, sigma_obs, time_steps)
    series = series + noise
    plt.plot(series)
    plt.title('Sinusoidal Synthetic Time Series (1) with noise')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid()
    plt.show()
    plt.clf()

    df = pd.DataFrame(series, columns=['concerned'])
    df.to_csv('data/synth/sin1noise_history.csv', index_label='Month')


if __name__ == '__main__':
    main()

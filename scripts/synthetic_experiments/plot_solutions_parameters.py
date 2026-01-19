import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from abm4problemod.model import ATBCRModel, FJModel, BiasedAssimilationModel


def draw_brace_axcoords(ax, y0_data, y1_data, y_total, x_ax=0.05, width=0.03,
                        color='black', lw=2):
    # Normalized from data coordinates to 0-1
    y0 = y0_data / y_total
    y1 = y1_data / y_total
    y_mid = (y0 + y1) / 2

    xs = [x_ax + width, x_ax, x_ax, x_ax, x_ax + width]
    ys = [y0, y0, y_mid, y1, y1]

    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round',
            transform=ax.transAxes, clip_on=False)


def create_entry(folder_name, local_od, month_columns, constrained):
    solution = pd.read_csv(f'{folder_name}/front.csv').iloc[0]

    if local_od == 'bc':
        solution_df = pd.DataFrame(columns=month_columns,
                                   index=[r'$\mu$', r'$\varepsilon$'])
    elif local_od == 'atbcr':
        solution_df = pd.DataFrame(columns=month_columns,
                                   index=[r'$\mu$', r'$\varepsilon$',
                                          r'$\vartheta$'])
    elif local_od == 'ab':
        solution_df = pd.DataFrame(columns=month_columns,
                                   index=[r'$\mu$', r'$\varepsilon$',
                                          r'$\gamma$'])
    elif local_od == 'ba':
        solution_df = pd.DataFrame(columns=month_columns, index=[r'$b$'])
    else:
        solution_df = pd.DataFrame(columns=month_columns, index=[r'$\xi$'])

    for i, month in enumerate(month_columns):
        if local_od == 'bc':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}']]
        elif local_od == 'atbcr':
            a = solution[f'threshold_bc_month_{i}']
            if constrained:
                b = solution[f'threshold_pol_month_{i}']
            else:
                b = solution[f'threshold_pol_month_{i}']
                b = a + (1 - a) * b
            solution_df[month] = [solution[f'convergence_month_{i}'], a, b]

        elif local_od == 'ab':
            solution_df[month] = [solution[f'convergence_month_{i}'],
                                  solution[f'threshold_bc_month_{i}'],
                                  solution[f'gamma_month_{i}']]
        elif local_od == 'ba':
            solution_df[month] = [solution[f'bias_month_{i}']]
        else:
            solution_df[month] = [solution[f'susceptibility_month_{i}']]

    return solution_df


def plot_models_param_scales(models, lims, output_file, colorbar=True,
                             title=None):
    # Palette for model names
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palette[1], palette[3] = palette[3], palette[1]
    palette = list(reversed(palette[2:7]))

    # Global dimensions
    n_meses = max(df.shape[1] for df in models.values())
    n_total_params = sum(df.shape[0] for df in models.values())

    fig, ax = plt.subplots(
        figsize=(1.2 * n_meses + 3, 0.6 * n_total_params + 2))
    cmap = plt.cm.RdYlGn
    y_start = 0  # vertical position where to start drawing

    # Repeat colors if there are more models than colors
    model_colors = palette * ((len(models) // len(palette)) + 2)

    for m_idx, ((nombre, df), color_model) in enumerate(
            zip(models.items(), model_colors)):
        n_params = df.shape[0]

        # Draw cells
        for i, (param, row) in enumerate(reversed(list(df.iterrows()))):
            y = y_start + i

            if param == r"$\vartheta$":
                vmin, vmax = lims[param][1], lims[param][0]
            else:
                vmin, vmax = lims[param]

            data_norm = (row.values - vmin) / (vmax - vmin)
            data_norm = np.clip(data_norm, 0, 1)

            for j, val_norm in enumerate(data_norm):
                rect = plt.Rectangle((j, y_start + i), 1, 1,
                                     color=cmap(val_norm))
                ax.add_patch(rect)

                # Contrast of text color
                r, g, b, _ = cmap(val_norm)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.5 else 'black'

                ax.text(j + 0.5, y_start + i + 0.5, f"{row.values[j]:.2f}",
                        ha="center", va="center", fontsize=15,
                        color=text_color, rotation=40)
            # Name of the parameter to the left of the cell
            ax.text(-0.2, y + 0.5, param, ha='right', va='center', fontsize=16,
                    color='black')

        # Name of the model
        if nombre == 'bc':
            nombre = 'dw'

        ax.text(-1, y_start + n_params / 2, nombre.upper(), ha="center",
                va="center", fontsize=18, weight="bold", color=color_model,
                rotation=90)

        # Draw brace for the model
        draw_brace_axcoords(ax, y_start, y_start + n_params,
                            y_total=n_total_params, x_ax=-0.04, width=0.03,
                            color='black', lw=2)

        y_start += n_params

        # Horizontal separator line between models
        if m_idx < len(models) - 1:
            ax.hlines(y=y_start, xmin=0, xmax=n_meses, color='white',
                      linewidth=8)

    # Adjust axes
    ax.set_xlim(0, n_meses)
    ax.set_ylim(0, n_total_params)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(n_meses) + 0.5)
    first_model = next(iter(models.values()))
    ax.set_xticklabels(first_model.columns, rotation=45, fontsize=15)
    ax.xaxis.tick_bottom()

    ax.set_yticks([])

    # Cell borders
    ax.set_xticks(np.arange(n_meses), minor=True)
    ax.set_yticks(np.arange(n_total_params), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05,
                            pad=0.02)

        # Replace ticks with 'min' and 'max'
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['min', 'max'], fontsize=15)

        # Título de la colorbar
        cbar.set_label("Parameter intensity", rotation=270, labelpad=0,
                       fontsize=15)

    if title is not None:
        plt.title(title, fontsize=18)

    plt.savefig(output_file, bbox_inches="tight")
    plt.clf()


def main():
    # Calibration alternatives
    local_ods = {'bc': ATBCRModel, 'atbcr': ATBCRModel, 'fj': FJModel,
                 'ab': ATBCRModel, 'ba': BiasedAssimilationModel}

    algorithm = 'DE'
    targets = ['stable', 'sin1', 'sin1noise']
    concern_thresholds = [0.6, 0.75, 0.9]

    daily_factor = 450
    max_ev = 30000
    pop_size = 100
    mc = 20

    month_columns = ['Month1', 'Month2', 'Month3', 'Month4', 'Month5',
                     'Month6', 'Month7', 'Month8', 'Month9', 'Month10',
                     'Month11', 'Month12', 'Month13', 'Month14', 'Month15']

    lims = {r"$\mu$": (0.0, 0.5), r"$\varepsilon$": (0.0, 1.0),
            r"$\vartheta$": (0.0, 1.0), r"$\gamma$": (0.1, 2.0),
            r"$b$": (0.1, 100.0), r"$\xi$": (0.1, 1.0)}

    for target in targets:
        for concern_threshold in concern_thresholds:
            results_dir = f'results/synth/summary_calibration_results/Concern{concern_threshold}/uncons_'
            os.makedirs(results_dir, exist_ok=True)

            models = {}

            for local_od in local_ods.keys():
                folder_name = f'results/synth/{target}_history/Concern{concern_threshold}/uncons_{local_od}_{daily_factor}_{mc}_{max_ev}_{pop_size}'

                if os.path.exists(folder_name):
                    solution_df = create_entry(folder_name, local_od,
                                               month_columns, False)

                    models[local_od] = solution_df

            # Reverse the order of models for better visualization
            models = dict(reversed(list(models.items())))
            output_file = f'results/summary_calibration_results/parameters_{target}_{concern_threshold}.pdf'
            title = f"Concern threshold = {concern_threshold}"

            colorbar = concern_threshold == 0.9
            plot_models_param_scales(models, lims, output_file, colorbar,
                                     title)


if __name__ == '__main__':
    main()

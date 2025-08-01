# Opinion Dynamics with Highly Oscillating Opinions

Supplementary material with the code and data used in the paper
***Opinion Dynamics with Highly Oscillating Opinions***,
co-authored by Víctor A. Vargas-Pérez, Jesús Giráldez-Cru, and Oscar Cordón.

## Experiments pipeline

The folder `scripts` contains the code used to run each step of the
experimentation. The pipeline is divided into the following steps:

1. 'create_social_networks.py': creates the social networks used in the
   experiments. In particular, since results are similar with the different
   topologies, we choose the scale-free network with <k> ~ 6. The nodes of
   these networks represent the agents, i.e., the respondents in '
   data/mig/mig01.csv'.

### Generate calibration results (`experiments` folder)

2. `run_degroot_model.py`: runs the DeGroot model.
3. `run_de_calibration.py`: runs the DE calibration of the selected Opinion
   Dynamics (OD) model.
4. `run_lshade_calibration.py`: runs the L-SHADE (or SHADE, depending on the
   arguments) calibration of the selected OD model.
5. `run_pso_calibration.py`: runs the PSO calibration of the selected OD

### Process calibration results (`process_results` folder)

6. `plot_all_calibration_fitness_evolution.py`: generates the fitness evolution
   plots for all the calibration results.
7. `plot_all_calibration_solutions_outputs.py`: generates the concern plots for
   all the calibration results.
8. `plot_all_calibration_comparison_concern.py`: generate a concern plot per OD
   model and concern threshold. The idea is to compare the results of the
   different Evolutionary Algorithms for each problem.
9. `process_all_calibration_results_csv.py`: generates the CSV files with all
   the calibration results + DeGroot model results.

### Additional scripts for paper-oriented results (`organize_results` folder)

10. `plot_fitness_evolution_comparison.py`: generates the fitness evolution
    comparison plots for ATBCR-09 models
11. `plot_concern_best_models.py`: generates a concern plot per threshold with
    the best models solutions.
12. `plot_concern_best_ATBCR.py`: generates the concern evolution of the best
    ATBCR-09 model with parameters values.
13. `build_mean_std_csv.py`: generates CSV files with the mean and std. dev. of
    the calibration results as shown in the paper.
14. `plot_initial_opinions.py`: generate the illustrative figure of the initial
    opinions generation.

## License

Read [LICENSE](./LICENSE).
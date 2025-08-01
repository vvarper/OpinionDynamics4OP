import os

import networkx as nx
import pandas as pd


def main():
    mean_degrees = [3, 6, 10]
    seeds = [37, 25, 42]
    problem = 'mig'

    ## 1. Topologies generation ###############################################

    n = len(pd.read_csv(f'data/{problem}/{problem}01.csv'))

    # Erdos Renyi topologies
    p_er = [d / n for d in mean_degrees]
    G_erdos = []

    for p, seed in zip(p_er, seeds):
        G_erdos.append(nx.erdos_renyi_graph(n=n, p=p, seed=seed))

    # Newman-Watts-Strogatz topologies with p=0.1
    k_list = mean_degrees
    p_newman = 0.1
    G_newman = []

    for k, seed in zip(k_list, seeds):
        G_newman.append(
            nx.watts_strogatz_graph(n=n, k=k, p=p_newman, seed=seed))

    # Barabasi-Albert topologies
    m_list = [d // 2 for d in mean_degrees]
    G_barabasi = []

    for m, seed in zip(m_list, seeds):
        G_barabasi.append(nx.barabasi_albert_graph(n=n, m=m, seed=seed))

    ## 3. Save graphs as GML files ############################################
    base_folder = f'data/{problem}/social_networks/'
    os.makedirs(os.path.dirname(base_folder), exist_ok=True)

    for i in range(len(seeds)):
        nx.write_gml(G_erdos[i],
                     f'{base_folder}erdos_{mean_degrees[i]:.4f}.gml')
        nx.write_gml(G_newman[i],
                     f'{base_folder}newman_{k_list[i]}_{p_newman}.gml')
        nx.write_gml(G_barabasi[i], f'{base_folder}barabasi_{m_list[i]}.gml')


if __name__ == '__main__':
    main()

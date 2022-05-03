""" This code has functions that calculate the fairness values of a given district plan """
""" The main() function prints the fairness values of a particular district plan located by its filename """

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from gurobipy import *
import itertools
import cplex
# print cplex.__version__
from cplex.exceptions import CplexSolverError
from cplex.callbacks import MIPInfoCallback
from cplex.callbacks import BranchCallback
from cplex.callbacks import NodeCallback
from cplex.callbacks import IncumbentCallback
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
import sys
import json
import random
from heapq import nlargest
import math
import operator
import bisect
import maxcardinalitymatching
from functools import reduce 
import cmd
import generate_plans

def plot_vote_seat_curve(z_k, graph):
    K = len(z_k)
    total_voted = sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in graph)
    curr_net_vote_share = float(sum(graph.nodes[i]['p_dem']for i in graph))/total_voted
    curr_net_seat_share = float(len([k for k in z_k if sum(graph.nodes[i]['p_dem']for i in z_k[k]) >= sum(graph.nodes[i]['p_rep']for i in z_k[k])]))/K
    # curr_net_seat_share = 4
    original_net_vote_share = curr_net_vote_share
    original_net_seat_share = curr_net_seat_share
    voted_k = {k: sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in z_k[k]) for k in range(1,K+1)}
    pdem_levels_k = {k: float(sum(graph.nodes[i]['p_dem'] for i in z_k[k]))/sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in z_k[k]) for k in range(1,K+1)}
    original_pdem_levels_k = pdem_levels_k.copy()

    voteshare_list = [0,1]
    seatsare_list = [0,1]
    seat_share_dict = {0:0, 1:1}
    pdem_levels_lessthanhalf_k = {k: pdem_levels_k[k] for k in pdem_levels_k if pdem_levels_k[k] < 0.5}
    while (len(pdem_levels_lessthanhalf_k) != 0):
        # print "< half:",pdem_levels_lessthanhalf_k
        next_k = max(iter(pdem_levels_lessthanhalf_k.items()), key=operator.itemgetter(1))[0]
        fraction_incremented = float(0.5 - pdem_levels_lessthanhalf_k[next_k])
        pdem_levels_k = {k: min(pdem_levels_k[k] + fraction_incremented,1) for k in range(1,K+1)}
        curr_net_vote_share = float(sum(pdem_levels_k[k]*voted_k[k] for k in range(1,K+1)))/total_voted
        curr_net_seat_share += float(1)/K
        # curr_net_seat_share += 1
        # print curr_net_vote_share,fraction_incremented, curr_net_seat_share
        seat_share_dict[curr_net_vote_share] = curr_net_seat_share
        seatsare_list.append(curr_net_seat_share)
        voteshare_list.append(curr_net_vote_share)
        pdem_levels_lessthanhalf_k = {k: pdem_levels_lessthanhalf_k[k] + fraction_incremented for k in pdem_levels_lessthanhalf_k}
        pdem_levels_lessthanhalf_k.pop(next_k)

    # print [i*total_voted for i in voteshare_list]

    pdem_levels_k = original_pdem_levels_k.copy()
    curr_net_vote_share = original_net_vote_share
    curr_net_seat_share = original_net_seat_share
    pdem_levels_morethan_k = {k: pdem_levels_k[k] for k in pdem_levels_k if pdem_levels_k[k] >= 0.5}
    while (len(pdem_levels_morethan_k) != 0):
        # print "> half:", pdem_levels_morethan_k
        next_k = min(iter(pdem_levels_morethan_k.items()), key=operator.itemgetter(1))[0]
        fraction_incremented = float(pdem_levels_morethan_k[next_k] - 0.5)
        pdem_levels_k = {k: max(pdem_levels_k[k] - fraction_incremented, 0) for k in range(1,K+1)}
        # print pdem_levels_k
        curr_net_vote_share = float(sum(pdem_levels_k[k]*voted_k[k] for k in range(1,K+1)))/total_voted
        seatsare_list.append(curr_net_seat_share)
        seat_share_dict[curr_net_vote_share] = curr_net_seat_share
        curr_net_seat_share -= float(1)/K
        # curr_net_seat_share -= 1
        # print curr_net_vote_share,fraction_incremented, curr_net_seat_share, pdem_levels_morethan_k
        voteshare_list.append(curr_net_vote_share)
        pdem_levels_morethan_k = {k: pdem_levels_morethan_k[k] - fraction_incremented for k in pdem_levels_morethan_k}
        pdem_levels_morethan_k.pop(next_k)

    voteshare_list = sorted(voteshare_list)
    seatsare_list = sorted(seatsare_list)
    flipped_voteshare_list = [1 - i for i in voteshare_list]
    flipped_seatsare_list = [1-i for i in seatsare_list]

    """ Plotting the graph """
    # print("vote shares:", voteshare_list)
    # print("seat shares:", seatsare_list)
    # print("flipped vote shares:", flipped_voteshare_list)
    # print("flipped seat shares:", flipped_seatsare_list)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    plt.grid()
    demplot, = plt.step(voteshare_list, seatsare_list, where = 'post', color = 'b', label = 'Democrats')

    repplot, = plt.step(flipped_voteshare_list, flipped_seatsare_list, where = 'post', color = 'r', linestyle='--', label = 'Republicans')
    # demplot, = plt.step(voteshare_list, seatsare_list, where = 'post', color = 'b', label = 'Party A')
    # repplot, = plt.step(flipped_voteshare_list, flipped_seatsare_list, where = 'post', color = 'r', linestyle='--', label = 'Party B')
    # plt.axes()
    plt.xlim([0, 1.05])
    # plt.xlabel('Average vote share across all districts', fontsize = 20)
    # plt.ylabel('Seat share (fraction of districts won)', fontsize = 20)
    # plt.scatter(.55, .5, s=50, color = 'b')
    # plt.scatter(.45, .5, s=50, color = 'r')
    plt.legend(handles = [demplot, repplot], loc=2, fontsize = 15)
    plt.show()


def compute_partisan_symmetry_with_formula(z_k, graph):
    K = len(z_k)
    u_k_dict = {k: float(sum(graph.nodes[i]['p_rep'] for i in z_k[k]))/sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
    u_k_list = sorted([u_k_dict[k] for k in u_k_dict], reverse = 1)

    v_j = {j: float(sum(abs(min(1,max(0, u_k_dict[k] - u_k_list[j-1] + 0.5))) for k in z_k))/K for j in z_k}

    PA = float(sum(abs(v_j[j]+v_j[K-j+1] - 1) for j in z_k))/K

    return PA


def evaluate_effgap(z_k, graph):
    z_k_copy = z_k.copy()
    total_voted = 0
    net_wasted_votes = 0
    for k in z_k:
        rep_pop = sum(graph.nodes[i]['p_rep'] for i in z_k[k])
        dem_pop = sum(graph.nodes[i]['p_dem'] for i in z_k[k])
        voted_pop = sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k[k])

        if rep_pop > dem_pop:
            who_won = 'R'
            dem_wasted_votes = dem_pop
            rep_wasted_votes = rep_pop - voted_pop*0.5
        else:
            who_won = 'D'
            dem_wasted_votes = dem_pop - voted_pop*0.5
            rep_wasted_votes = rep_pop
        total_voted += voted_pop
        net_wasted_votes += dem_wasted_votes-rep_wasted_votes
    eff_gap = float(net_wasted_votes)/float(total_voted)
    return eff_gap


def evaluate_number_of_cmptitv_dists(z_k, graph, margin_of_victory):
    pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
    pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
    pop_voted_k = {k: pop_rep_k[k]+pop_dem_k[k]  for k in z_k}
    return [k for k in z_k if float(abs(pop_rep_k[k]-pop_dem_k[k]))/pop_voted_k[k] <= margin_of_victory]


def evaluate_compactness_sumdist(z_k, graph, d_matrix):
    district_centers_k = {}
    K = len(z_k)
    for k in range(1,K+1):
        distance_dict_i = {i: sum(graph.nodes[j]['pop']*d_matrix[i][j]**2 for j in z_k[k]) for i in z_k[k]}

        district_centers_k[k] = min(list(distance_dict_i.items()), key=lambda x: x[1])[0]

    return sum(sum(graph.nodes[j]['pop']*d_matrix[district_centers_k[k]][j]**2 for j in z_k[k]) for k in z_k), district_centers_k


def contiguity_check(z_k,graph):
    K = len(z_k)
    for k in z_k:
        sub_graph = nx.Graph()
        sub_graph = graph.subgraph(z_k[k])
        if not nx.is_connected(sub_graph):
            return k,0

    return K, 1


def evaluate_population_balance(pop_k,graph):
    K = len(pop_k)
    P_bar = sum(pop_k[k] for k in pop_k)/len(pop_k)
    return float(max(abs(pop_k[k] - P_bar) for k in range(1,K+1)))/float(P_bar)


def evaluate_compactness_edgecuts(z_i, graph):
    weight_of_cuts = 0
    for (i,j) in graph.edges():
        if z_i[i] != z_i[j]:
            # weight_of_cuts += 1
            weight_of_cuts += graph[i][j]['bndry']
    return weight_of_cuts


def evaluate_max_margin(z_k, graph):
    pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
    pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
    return max([float(abs(pop_rep_k[k]-pop_dem_k[k]))/(pop_rep_k[k]+pop_dem_k[k]) for k in z_k])


def evaluate_strong_dists(z_k, graph, margin_of_victory):
    pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
    pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
    pop_voted_k = {k: pop_rep_k[k]+pop_dem_k[k]  for k in z_k}
    return [k for k in z_k if float(pop_rep_k[k]-pop_dem_k[k])/pop_voted_k[k] >= margin_of_victory], [k for k in z_k if float(pop_dem_k[k]-pop_rep_k[k])/pop_voted_k[k] >= margin_of_victory]


def print_metrics(z_k_tract, z_i_tract, tract_graph, d_matrix):
    K = len(z_k_tract)
    P_bar = int(sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes())/K)
    print("Metrics:")
    print("Overall R vote-share:", float(sum(tract_graph.nodes[i]['p_rep'] for i in tract_graph.nodes()))/(sum(tract_graph.nodes[i]['p_rep']+tract_graph.nodes[i]['p_dem'] for i in tract_graph.nodes())))

    pop_k = {k: sum(tract_graph.nodes[i]['pop'] for i in z_k_tract[k]) for k in z_k_tract}
    print("Population balance:", evaluate_population_balance(pop_k,tract_graph), [pop_k[k] for k in pop_k])

    compactness = evaluate_compactness_sumdist(z_k_tract, tract_graph, d_matrix)
    print("Compactness:", compactness)
    print("Efficiency gap:", evaluate_effgap(z_k_tract, tract_graph))
    print("Partisan asymmetry:", compute_partisan_symmetry_with_formula(z_k_tract, tract_graph))
    max_margin =  evaluate_max_margin(z_k_tract, tract_graph)
    print("Max margin (cmpttv):", max_margin)
    print("No. of 10% cmpttv districts:",len(evaluate_number_of_cmptitv_dists(z_k_tract, tract_graph, 0.1)))
    print("Cmpttv districts:", evaluate_number_of_cmptitv_dists(z_k_tract, tract_graph, 0.1))
    print("Strong D districts:", evaluate_strong_dists(z_k_tract, tract_graph, 0.1)[1])
    print("Simple D districts:", evaluate_strong_dists(z_k_tract, tract_graph, 0)[1])
    print("Strong R districts:", evaluate_strong_dists(z_k_tract, tract_graph, 0.1)[0])
    print("Simple R districts:", evaluate_strong_dists(z_k_tract, tract_graph, 0)[0])
    for k in z_k_tract:
        dem_vote_share = sum(tract_graph.nodes[i]['p_dem'] for i in z_k_tract[k])/sum(tract_graph.nodes[i]['p_dem']+tract_graph.nodes[i]['p_rep'] for i in z_k_tract[k])
        margin = abs(0.5-dem_vote_share)*2
        dem_pop = sum(tract_graph.nodes[i]['p_dem'] for i in z_k_tract[k])
        rep_pop = sum(tract_graph.nodes[i]['p_rep'] for i in z_k_tract[k])
        voted_pop = sum(tract_graph.nodes[i]['p_rep']+tract_graph.nodes[i]['p_dem'] for i in z_k_tract[k])
        dem_wasted_votes, rep_wasted_votes = 0, 0
        if dem_vote_share > 0.5:
            who_won = 'D'
            dem_wasted_votes = dem_pop - voted_pop*0.5
            rep_wasted_votes = rep_pop
        else:
            who_won = 'R'
            dem_wasted_votes = dem_pop
            rep_wasted_votes = rep_pop - voted_pop*0.5

        stuff_to_print = [k,pop_k[k], abs(P_bar-pop_k[k]), abs(P_bar-pop_k[k])/P_bar, margin, who_won, dem_wasted_votes, rep_wasted_votes, dem_vote_share, 1-dem_vote_share]
        print(', '.join(str(i) for i in stuff_to_print))

    return evaluate_effgap(z_k_tract, tract_graph), compute_partisan_symmetry_with_formula(z_k_tract, tract_graph), evaluate_max_margin(z_k_tract, tract_graph), evaluate_compactness_edgecuts(z_i_tract, tract_graph)


def read_plan_from_file(filename):
    z_k = {k:[] for k in range(1,K+1)}
    z_i = {}
    with open(filename+'.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rowlist = [row for row in spamreader]
        edgelist = []
        for i in range(1,len(rowlist)):
            if rowlist[i] != []:
                # print(rowlist[i])
                z_k[int(rowlist[i][1])].append(int(rowlist[i][0]))
                z_i[int(rowlist[i][0])] = int(rowlist[i][1])

    return z_i, z_k


def plot_plan(graph, z_k, z_i):
    K = len(z_k)
    pos = {i:(graph.nodes[i]['x'], graph.nodes[i]['y']) for i in graph.nodes()}
    color_k = {1:'b', 2:'g', 3:'r', 4: 'c', 5:'m', 6:'y', 7:'k', 8: 'gray'}
    for k in range(1,K+1):
        nx.draw_networkx_nodes(graph,pos, nodelist=z_k[k], node_color=color_k[k], node_size = 20)
        for (i,j) in graph.edges():
            if z_i[i] == k and z_i[j] == k:
                nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = color_k[k])

    for (i,j) in graph.edges():
        if z_i[i] != z_i[j]:
            nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = 'k', style = 'dotted')

    plt.axis('off')
    plt.show()


def plot_graph(graph):
    pos = {i:(graph.nodes[i]['x'], graph.nodes[i]['y']) for i in graph.nodes()}
    nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(), node_color='k', node_size = 20)
    for (i,j) in graph.edges():
        nx.draw_networkx_edges(graph, pos, edgelist = [(i,j)], edge_color = 'k')

    plt.axis('off')
    plt.show()



def main():
    tract_graph, d_matrix = generate_plans.input.read_data_tract()
    global K
    K = 8
    # filename = 'results/Most_fair_maps/most_equitable_map'
    filename = 'results/Most_fair_maps/most_symmetric_map'
    # filename = 'results/Most_fair_maps/most_cmpttv_map'
    plot_graph(tract_graph)
    z_i, z_k = read_plan_from_file(filename)
    print_metrics(z_k, z_i, tract_graph, d_matrix)
    plot_plan(tract_graph, z_k, z_i)

if __name__ == "__main__":
    main()

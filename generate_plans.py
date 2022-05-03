"""
This code generates district plans by solving one of four problems: single-objectivee Compact Distriting Problem

"""
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from gurobipy import *
import itertools
import cplex
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
import compute_metrics
import os
# import pickle


class input:
    def read_data_tract():
        """ Reads the inputs for the Wisconsin instance - adjacency list, population, 2012 and 2016 presidential election Results, and the geospatial coordinates of every census tract """
        G = nx.Graph()

        with open('Datasets/WI_adjacency_tracts.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rowlist = [row for row in spamreader]
            edgelist = []
            bndry = {}
            for i in range(len(rowlist)):
                if int(rowlist[i][0]) != int(rowlist[i][1]):
                    edgelist.append((int(rowlist[i][1]),int(rowlist[i][2])))
                    bndry[int(rowlist[i][1]),int(rowlist[i][2])] = float(rowlist[i][3])

        G.add_edges_from(edgelist)
        for (i,j) in bndry:
            G[i][j]['bndry'] = bndry[(i,j)]

        with open('Datasets/WI_2010census_population_tracts.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            pop = {}
            for i in range(len(row_list)):
                if i != 0 and int(row_list[i][0]) in G.nodes():
                    pop[int(row_list[i][0])] = int(float(row_list[i][1]))

        with open('Datasets/WI_2016presidential_tracts.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            p_dem_2016, p_rep_2016 = {}, {}
            for i in range(len(row_list)):
                if i != 0 and int(row_list[i][0]) in G.nodes():
                    p_dem_2016[int(row_list[i][0])] = int(float(row_list[i][1]))
                    p_rep_2016[int(row_list[i][0])] = int(float(row_list[i][2]))

        with open('Datasets/WI_2012presidential_tracts.csv') as csvfile:
        # with open('illinois_county_political_population.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            p_dem_2012, p_rep_2012 = {}, {}
            for i in range(len(row_list)):
                if i != 0 and int(row_list[i][0]) in G.nodes():
                    p_dem_2012[int(row_list[i][0])] = int(float(row_list[i][1]))
                    p_rep_2012[int(row_list[i][0])] = int(float(row_list[i][2]))

        with open('Datasets/WI_coordinates_tracts.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            x_coord, y_coord = {}, {}
            for i in range(len(row_list)):
                x_coord[int(row_list[i][0])] = float(row_list[i][1])
                y_coord[int(row_list[i][0])] = float(row_list[i][2])

        for i in G.nodes():
            G.nodes[i]['x'] = x_coord[i]
            G.nodes[i]['y'] = y_coord[i]
            G.nodes[i]['pop'] = pop[i]
            G.nodes[i]['p_dem'] = (p_dem_2012[i] + p_dem_2016[i])/2
            G.nodes[i]['p_rep'] = (p_rep_2012[i] + p_rep_2016[i])/2

        # nreps = sum(G.nodes[i]['p_rep'] for i in G.nodes())
        # ndems = sum(G.nodes[i]['p_dem'] for i in G.nodes())
        # print("Wisconsin is",float(nreps)*1000/float(nreps+ndems),"% Republican")

        d_matrix = dict(nx.all_pairs_shortest_path_length(G)) # the distance matrix between every pair of census tracts

        return G, d_matrix


class coarsen:

    def solve_maximal_matching_problem(input_graph):
        graph_copy = nx.Graph()
        graph_copy.add_edges_from([(i,j) for (i,j) in input_graph.edges()])
        for (i,j) in graph_copy.edges():
            graph_copy[i][j]['weight'] = input_graph[i][j]['weight']
        matching_edges_list = []
        iterations_so_far = 0
        while(len(graph_copy.edges()) > 0):
            edge_weight_dict = {(i,j): graph_copy[i][j]['weight'] for (i,j) in graph_copy.edges()}
            best_edge = min(iter(edge_weight_dict.items()), key=operator.itemgetter(1))[0]
            matching_edges_list.append(best_edge)

            graph_copy.remove_nodes_from([best_edge[0],best_edge[1]])
            iterations_so_far += 1

        return matching_edges_list


    def solve_maximum_matching_problem(input_graph):
        graph_copy = nx.Graph()
        graph_copy = input_graph.copy()

        matching_edges_list = []

        edge_weight_dict = {(i,j): graph_copy[i][j]['weight'] for (i,j) in graph_copy.edges()}
        sorted_edges = [item[0] for item in sorted(list(edge_weight_dict.items()), key=operator.itemgetter(1))]

        def maxcardmatching_given_edgepointer(pointer):
            subgraph = nx.Graph()
            subgraph.add_edges_from(sorted_edges[:pointer])
            maxcardinality_matching_subgraph = [(key,val) for (key,val) in list(maxcardinalitymatching.matching(subgraph,{}).items())]
            maxcardinality_matching_subgraph = set(tuple(sorted(l)) for l in maxcardinality_matching_subgraph)
            maxcardinality_matching_subgraph_size = len(maxcardinality_matching_subgraph)
            return maxcardinality_matching_subgraph, maxcardinality_matching_subgraph_size

        total_n_edges = len(graph_copy.edges())
        maxcardinality_matching_size = maxcardmatching_given_edgepointer(total_n_edges)[1]
        for edge_pointer in sorted(list(range(total_n_edges)),reverse=True):
            maxcardinality_matching_subgraph, maxcardinality_matching_subgraph_size = maxcardmatching_given_edgepointer(edge_pointer)
            if maxcardinality_matching_subgraph_size < maxcardinality_matching_size:
                break

        matching_edges_list = maxcardmatching_given_edgepointer(edge_pointer+1)[0]

        return matching_edges_list


    def coarsen_multilevel(graph):
        """ Inputs a graph and outputs a coarsened graph after L levels of coarsening  """

        """ Initiating level 0 instance """
        level_graph = {}
        level_graph[0] = graph.copy()

        d_matrix_level_l = {}
        d_matrix_level_l[0] = dict(nx.all_pairs_shortest_path_length(level_graph[0]))

        print("Level 0 graph: Number of edges:",len(level_graph[0].edges()), "Number of nodes:",len(level_graph[0].nodes()))

        level_l_to_l_plus_1 = {} # stores index of a coarsened unit in level l+1 for each unit in level l for l = 0,1,...,L-1
        level_l_plus_1_to_l = {} # stores indices of the uncoarsened units in level l for each coarse unit in level l+1 for l = 0,1,...,L-1

        """ Aggregating from level 0 """
        for l in range(1,L+1):
            pop_dict = {i: level_graph[l-1].nodes[i]['pop'] for i in level_graph[l-1].nodes()} # population of every unit in level l-1
            print("Population range in level",l-1,"is",min(pop_dict[i] for i in pop_dict),"-",max(pop_dict[i] for i in pop_dict),"average:",float(sum(pop_dict[i] for i in pop_dict))/len(pop_dict))
            edge_weights = {(i,j): pop_dict[i]+pop_dict[j] for (i,j) in level_graph[l-1].edges()} # edge weights given by the sum of the populations of the two units

            for (i,j) in level_graph[l-1].edges():
                level_graph[l-1][i][j]['weight'] = (edge_weights[i,j])

            if matching_strategy == 'ML':
                matchings = coarsen.solve_maximal_matching_problem(level_graph[l-1]) # find a maximal matching using greedy algorithm
            else:
                matchings = coarsen.solve_maximum_matching_problem(level_graph[l-1]) # find a maximum matching using Edmonds' algorithm

            matched_nodes = [i for i in level_graph[l-1].nodes() if any(i in edge for edge in matchings)] # set of matched units
            unmatched_nodes = [i for i in level_graph[l-1].nodes() if i not in matched_nodes] # set of unmatched units
            print("Number of matched units:", len(matched_nodes))
            print("Number of unmatched units:", len(unmatched_nodes))

            """ Creating the coarse units in graph level l using the matched units """
            level_graph[l] = nx.Graph()
            level_l_to_l_plus_1[l-1] = {}
            level_l_plus_1_to_l[l] = {}
            node_counter = 0
            for (i,j) in matchings:
                level_l_to_l_plus_1[l-1][i] = node_counter
                level_l_to_l_plus_1[l-1][j] = node_counter
                level_l_plus_1_to_l[l][node_counter] = [i,j]

                level_graph[l].add_node(node_counter, pop = level_graph[l-1].nodes[i]['pop'] + level_graph[l-1].nodes[j]['pop'], p_dem = level_graph[l-1].nodes[i]['p_dem'] + level_graph[l-1].nodes[j]['p_dem'], p_rep = level_graph[l-1].nodes[i]['p_rep'] + level_graph[l-1].nodes[j]['p_rep'],\
                                    x =  float(level_graph[l-1].nodes[i]['x']+level_graph[l-1].nodes[j]['x'])/2, y =  float(level_graph[l-1].nodes[i]['y']+level_graph[l-1].nodes[j]['y'])/2)
                node_counter += 1

            """ Creating the coarse units in graph level l using the unmatched units """
            for i in unmatched_nodes:
                level_l_to_l_plus_1[l-1][i] = node_counter
                level_l_plus_1_to_l[l][node_counter] = [i]
                level_graph[l].add_node(node_counter, pop = level_graph[l-1].nodes[i]['pop'], p_dem = level_graph[l-1].nodes[i]['p_dem'], p_rep = level_graph[l-1].nodes[i]['p_rep'], x = level_graph[l-1].nodes[i]['x'], y = level_graph[l-1].nodes[i]['y'])
                node_counter += 1

            """ Creating a edges in graph level l  """
            for (i,j) in level_graph[l-1].edges():
                upper_level_unit_of_i = level_l_to_l_plus_1[l-1][i]
                upper_level_unit_of_j = level_l_to_l_plus_1[l-1][j]
                if upper_level_unit_of_i != upper_level_unit_of_j:
                    level_graph[l].add_edge(upper_level_unit_of_i, upper_level_unit_of_j)
                    level_graph[l][upper_level_unit_of_i][upper_level_unit_of_j]['bndry'] = 0

            """ Updating the lengths of the shared boundaries in the coarse level graph """
            for (i,j) in level_graph[l-1].edges():
                upper_level_unit_of_i = level_l_to_l_plus_1[l-1][i]
                upper_level_unit_of_j = level_l_to_l_plus_1[l-1][j]
                if upper_level_unit_of_i != upper_level_unit_of_j:
                    level_graph[l][upper_level_unit_of_i][upper_level_unit_of_j]['bndry'] += level_graph[l-1][i][j]['bndry']

            # compute_metrics.plot_graph(level_graph[l])

            print("Number of edges in level", l ,"graph:",len(level_graph[l].edges()), "Number of nodes:",len(level_graph[l].nodes()))
            print("Heaviest unit population:", max([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]), "Lightest unit population:", min([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]), "Ratio of heaviest and lightest unit populations:", float(max([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]))/(min([level_graph[l].nodes[i]['pop'] for i in level_graph[l]])+1),"\n"),

            """ Creating the distance matric in the coarse level graph """
            d_matrix_level_l[l] = {}
            for i in level_graph[l]:
                d_matrix_level_l[l][i] = {}
                for j in level_graph[l]:
                    d_matrix_level_l[l][i][j] = min(sum(d_matrix_level_l[l-1][i_lower][j_lower] for j_lower in level_l_plus_1_to_l[l][j]) for i_lower in level_l_plus_1_to_l[l][i])

        return level_graph[L], level_graph[0], level_l_to_l_plus_1, level_l_plus_1_to_l, d_matrix_level_l, level_graph


class algorithm:
    def vickrey_initial_solution_heuristic(graph, d_matrix, start_time_initial_heuristic):
        # Inputs graph, and the distance matrix to find a feasible solution to the districting problem
        print("Starting to find an initial feasible solution")

        pop = {i: graph.nodes[i]['pop'] for i in list(graph.nodes())}
        iterations = 0
        while(1):
            z_k, z_i = {}, {}
            district_pop = {}
            unassigned_units = list(graph.nodes())
            """ Initial creation of districts """
            k = 1
            while(unassigned_units != []):
                if k not in z_k:
                    first_unit = random.choice(unassigned_units)
                    z_k[k] = [first_unit]
                    z_i[first_unit] = k
                    unassigned_units.remove(first_unit)
                    district_pop[k] = pop[first_unit]
                else:
                    candidate_units = list(set(unassigned_units)&set([j for i in z_k[k] for j in graph.neighbors(i) if pop[j]+district_pop[k] <= P_bar*(1+tau)]))
                    if candidate_units != []:
                        next_unit = random.choice(candidate_units)
                        z_k[k].append(next_unit)
                        z_i[next_unit] = k
                        unassigned_units.remove(next_unit)
                        district_pop[k] += pop[next_unit]
                    else:
                        k += 1

            least_pop =  min([val for key,val in list(district_pop.items())])
            most_pop =  max([val for key,val in list(district_pop.items())])
            # print least_pop, most_pop

            """ Iterative merging till K districts """
            while(len(z_k) > K):
                least_pop_distr = min(district_pop, key=district_pop.get)
                neighboring_distrs = list(set([z_i[j] for i in z_k[least_pop_distr] for j in graph.neighbors(i)]))
                least_pop_neighbor_distr = min({key: val for key,val in list(district_pop.items()) if key in neighboring_distrs and key != least_pop_distr}, key=district_pop.get)
                merged_distr = min(least_pop_neighbor_distr,least_pop_distr)
                other_distr = max(least_pop_neighbor_distr,least_pop_distr)
                z_k[merged_distr] = z_k[least_pop_neighbor_distr] + z_k[least_pop_distr]
                for i in z_k[least_pop_distr]:
                    z_i[i] = merged_distr
                for i in z_k[least_pop_neighbor_distr]:
                    z_i[i] = merged_distr
                district_pop[merged_distr] = district_pop[least_pop_neighbor_distr] + district_pop[least_pop_distr]
                z_k.pop(other_distr, None)
                district_pop.pop(other_distr, None)
            z_k_copy, z_i_copy = {}, {}
            k = 1
            for k_dash in z_k:
                z_k_copy[k] = z_k[k_dash]
                for i in z_k[k_dash]:
                    z_i_copy[i] = k
                k += 1
            z_k = z_k_copy.copy()
            z_i = z_i_copy.copy()

            """ Local search method to improve population balance"""
            print("Starting local search improvement")
            z_k, z_i, obj = algorithm.local_search_flip(z_k, z_i, graph, 'pop_bal', {})

            district_pop = {k: sum(pop[i] for i in z_k[k]) for k in z_k}
            least_pop =  min([val for key,val in list(district_pop.items())])
            most_pop =  max([val for key,val in list(district_pop.items())])

            if time.time() - start_time_initial_heuristic > initialheuristic_timelimit:
                return z_i, z_k

            feasibility_criteria = least_pop >= P_bar*(1-tau) and most_pop <= P_bar*(1+tau)
            if feasibility_criteria:
                if what_problem == 'EDP':
                    effgap = abs(compute_metrics.evaluate_effgap(z_k, graph))
                    feasibility_criteria = feasibility_criteria and abs(effgap) <= epsilon_EG
                elif what_problem == 'SDP':
                    partsanasymm = compute_metrics.compute_partisan_symmetry_with_formula(z_k, graph)
                    feasibility_criteria = feasibility_criteria and (partsanasymm <= epsilon_PA)
                elif what_problem == 'CmpttvDP':
                    obj_competitive = compute_metrics.evaluate_max_margin(z_k, graph)
                    feasibility_criteria = feasibility_criteria and (obj_competitive <= epsilon_cmpttv)
                if feasibility_criteria:
                    return z_i, z_k


    def get_warmsart_from_z(z_i, z_k, graph, d_matrix):
        if z_i == {}:
            return {}, {}
        warmstart_x = {}
        warmstart_dict = {}
        district_center = {}
        length = dict(nx.all_pairs_shortest_path_length(graph))
        for k in z_k:
            best_dist = 9999999999999999
            for i in z_k[k]:
                total_dist_i = sum(d_matrix[i][j] for j in z_k[k])
                if total_dist_i <= best_dist:
                    district_center[k] = i
                    best_dist = total_dist_i

        for i in list(graph.nodes()):
            for j in graph.nodes():
                if district_center[z_i[j]] == i:
                    warmstart_x["x_%i,%i"%(i,j)] = 1
                    warmstart_dict[(i,j)] = 1
                else:
                    warmstart_x["x_%i,%i"%(i,j)] = 0
                    warmstart_dict[(i,j)] = 0
        return warmstart_x, warmstart_dict


    def local_search_flip(z_k_initial, z_i_initial, graph, criteria, d_matrix):
        """ Inputs a district plan and performs flip-based local search to improve an pbjective function given by 'criteria' """
        """ The following functions efficiently recalculate the problem criteria after every flip iteration """

        def contiguity_check_after_move(z_k, unit, from_district, to_district, graph):
            flag = 0
            from_district_subgraph = graph.subgraph(z_k[from_district])
            from_district_subgraph.remove_nodes_from([unit])
            to_district_subgraph = graph.subgraph(z_k[to_district]+[unit])
            if len(from_district_subgraph.nodes()) !=0:
                if nx.is_connected(from_district_subgraph) + nx.is_connected(to_district_subgraph) == 2:
                    flag = 1
                else:
                    flag = 0
            else:
                flag = 0
            return flag

        def contiguity_check_after_move_neighbourhood(z_k, unit, from_district, to_district, graph):
            flag = 0
            unit_neighborhood_fromdistrict = list(set(graph.neighbors(unit)).intersection(set(z_k[from_district])))
            unit_neighborhood_fromdistrict_subgraph = graph.subgraph(unit_neighborhood_fromdistrict)
            if unit_neighborhood_fromdistrict != []:
                flag = nx.is_connected(unit_neighborhood_fromdistrict_subgraph)
            else:
                flag = 0

            return flag

        def population_balance_check(pop_k, unit, from_district, to_district, graph):
            flag = 1
            from_district_pop = pop_k[from_district] - graph.nodes[unit]['pop']
            to_district_pop = pop_k[to_district] + graph.nodes[unit]['pop']
            if from_district_pop >= pop_bal_lb and from_district_pop <= pop_bal_ub and to_district_pop >= pop_bal_lb and to_district_pop <= pop_bal_ub:
                flag = 1
            else:
                flag = 0
            return flag

        def evaluate_incremental_population_balance(pop_k, unit, from_district, to_district, graph):
            new_pop_k = pop_k.copy()
            new_pop_k[from_district] = new_pop_k[from_district] -  graph.nodes[unit]['pop']
            new_pop_k[to_district] = new_pop_k[to_district] +  graph.nodes[unit]['pop']
            return float(max(abs(new_pop_k[k] - P_bar) for k in range(1,K+1)))/float(P_bar)

        def evaluate_incremental_compactness_perimeter(z_k,current_compactness, unit, from_district, to_district, graph):
            final_compactness = current_compactness
            for j in set(graph.neighbors(unit)).intersection(set(z_k[from_district])):
                final_compactness += graph[unit][j]['bndry']
            for j in set(graph.neighbors(unit)).intersection(set(z_k[to_district])):
                final_compactness -= graph[unit][j]['bndry']

            return final_compactness

        def evaluate_incremental_compactness_sumdist(z_k, district_centers_k, current_compactness, unit, from_district, to_district,graph):
            final_sumdist = current_compactness
            district_centers_k_new = district_centers_k.copy()
            final_sumdist -= sum(graph.nodes[i]['pop']*d_matrix[district_centers_k[from_district]][i]**2 for i in z_k[from_district])
            final_sumdist -= sum(graph.nodes[i]['pop']*d_matrix[district_centers_k[to_district]][i]**2 for i in z_k[to_district])
            sumdists_in_fromdistr = {i: sum(graph.nodes[i]['pop']*d_matrix[i][j]**2 for j in z_k[from_district] if j!= unit) for i in z_k[from_district] if i != unit}
            new_fromdistrict_center = min(list(sumdists_in_fromdistr.items()), key=lambda x: x[1])[0]
            district_centers_k_new[from_district] = new_fromdistrict_center

            sumdists_in_todistr = {i: sum(graph.nodes[i]['pop']*d_matrix[i][j]**2 for j in z_k[to_district]+[unit]) for i in z_k[to_district]+[unit]}
            new_todistrict_center = min(list(sumdists_in_todistr.items()), key=lambda x: x[1])[0]
            district_centers_k_new[to_district] = new_todistrict_center
            final_sumdist += sum(graph.nodes[i]['pop']*d_matrix[district_centers_k_new[from_district]][i]**2 for i in z_k[from_district] if i != unit)

            final_sumdist += sum(graph.nodes[i]['pop']*d_matrix[district_centers_k_new[to_district]][i]**2 for i in z_k[to_district]+[unit])

            return district_centers_k_new, final_sumdist

        def evaluate_max_margin(z_k, unit, from_district, to_district, graph):
            pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
            pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}

            pop_rep_k[from_district] = pop_rep_k[from_district] - graph.nodes[unit]['p_rep']
            pop_dem_k[from_district] = pop_dem_k[from_district] - graph.nodes[unit]['p_dem']
            pop_rep_k[to_district] = pop_rep_k[to_district] + graph.nodes[unit]['p_rep']
            pop_dem_k[to_district] = pop_dem_k[to_district] + graph.nodes[unit]['p_dem']
            pop_voted_k = {k: pop_rep_k[k]+pop_dem_k[k]  for k in z_k}
            return max([float(abs(pop_rep_k[k]-pop_dem_k[k]))/(pop_voted_k[k]) for k in z_k])

        def evaluate_effgap_incremental(z_k, unit, from_district, to_district, graph):
            z_k_copy = z_k.copy()

            total_voted = 0
            net_wasted_votes = 0
            for k in z_k_copy:
                if k == from_district:
                    rep_pop = sum(graph.nodes[i]['p_rep'] for i in z_k_copy[k]) - graph.nodes[unit]['p_rep']
                    dem_pop = sum(graph.nodes[i]['p_dem'] for i in z_k_copy[k]) - graph.nodes[unit]['p_dem']
                    voted_pop = sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k_copy[k]) - (graph.nodes[unit]['p_rep']+graph.nodes[unit]['p_dem'])
                elif k == to_district:
                    rep_pop = sum(graph.nodes[i]['p_rep'] for i in z_k_copy[k]) + graph.nodes[unit]['p_rep']
                    dem_pop = sum(graph.nodes[i]['p_dem'] for i in z_k_copy[k]) + graph.nodes[unit]['p_dem']
                    voted_pop = sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k_copy[k]) + (graph.nodes[unit]['p_rep']+graph.nodes[unit]['p_dem'])
                else:
                    rep_pop = sum(graph.nodes[i]['p_rep'] for i in z_k_copy[k])
                    dem_pop = sum(graph.nodes[i]['p_dem'] for i in z_k_copy[k])
                    voted_pop = sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k_copy[k])

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

        def evaluate_passymm_incremental(z_k, unit, from_district, to_district, graph):

            z_k_copy_copy = {}
            for k in z_k:
                z_k_copy_copy[k] = []
                for i in z_k[k]:
                    z_k_copy_copy[k].append(i)
            z_k_copy_copy[from_district].remove(unit)
            z_k_copy_copy[to_district].append(unit)

            return compute_metrics.compute_partisan_symmetry_with_formula(z_k_copy_copy, graph)

        """ Initializing the algorithm """
        z_i_best = z_i_initial.copy()
        z_k_best = z_k_initial.copy()
        pop_k = {k: sum(graph.nodes[i]['pop'] for i in z_k_best[k]) for k in range(1,K+1)}

        if criteria == 'pop_bal':
            obj_best = compute_metrics.evaluate_population_balance(pop_k,graph)
        elif criteria == 'sumdist':
            obj_best, district_centers_k = compute_metrics.evaluate_compactness_sumdist(z_k_best, graph, d_matrix)
            print("starting obj:",obj_best)

        effgap_initial = compute_metrics.evaluate_effgap(z_k_best, graph)
        eff_gap = effgap_initial
        candidate_moves = []
        iteration = 0
        local_search_start_time = time.time()

        while((candidate_moves!=[] or iteration==0) and ((time.time()-local_search_start_time) <= initialheuristic_timelimit)):
            candidate_moves = []
            iteration += 1
            for i in list(graph.nodes()):
                 for k in set([z_i_best[j] for j in list(graph.neighbors(i)) if z_i_best[j]!= z_i_best[i]]):
                    z_i = z_i_best.copy()
                    z_i[i] = k

                    pop_bal = evaluate_incremental_population_balance(pop_k, i, z_i_best[i], k, graph)

                    feasibility_criteria = {criteria: contiguity_check_after_move_neighbourhood(z_k_best, i, z_i_best[i], k, graph)}
                    if criteria == 'sumdist':
                        feasibility_criteria['sumdist'] = feasibility_criteria[criteria] and (pop_bal < tau)
                        if what_problem == 'EDP':
                            eff_gap = evaluate_effgap_incremental(z_k_best, i, z_i_best[i], k, graph)
                            feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and (abs(eff_gap) <= abs(epsilon_EG))
                        elif what_problem == 'CmpttvDP':
                            max_margin = evaluate_max_margin(z_k_best, i, z_i_best[i], k, graph)
                            feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and max_margin <= epsilon_cmpttv
                        elif what_problem == 'SDP':
                            pasymm_incremental = evaluate_passymm_incremental(z_k_best.copy(), i, z_i_best[i], k, graph)
                            feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and pasymm_incremental <= epsilon_PA

                    if feasibility_criteria[criteria]:
                        if criteria == 'pop_bal':
                            obj = pop_bal
                        elif criteria== 'sumdist':
                            district_centers_k_modified, obj = evaluate_incremental_compactness_sumdist(z_k_best, district_centers_k, obj_best, i, z_i_best[i], k, graph)
                        if obj < obj_best:
                            candidate_moves.append(i)
                            i_best = i
                            k_best = k
                            from_dist = z_i_best[i_best]
                            z_k_best[k_best].append(i_best)
                            z_k_best[z_i_best[i_best]].remove(i_best)
                            pop_k[z_i_best[i_best]] = pop_k[z_i_best[i_best]] - graph.nodes[i_best]['pop']
                            pop_k[k_best] = pop_k[k_best] + graph.nodes[i_best]['pop']
                            z_i_best[i_best] = k_best
                            obj_best = obj
                            if criteria == 'sumdist':
                                print("iteration", iteration,": unit", i_best, "is moved from district", from_dist,"to district", k_best,"with current obj", obj_best)
                                district_centers_k = district_centers_k_modified.copy()
                            iteration += 1

        return z_k_best, z_i_best, obj_best


    def uncoarsen_with_local_search(z_i, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph_l, d_matrix_level_l):

        """ Uncoarsening in the decreasing order of levels from L through 0 """
        z_i_level_l = {L: z_i.copy()}
        z_k_level_l = {L: z_k.copy()}

        for l in sorted(list(range(0,L)),reverse = 1):
            print("Uncoarsening level",l)
            z_i_level_l[l] = {}
            z_k_level_l[l] = {}
            for i in z_i_level_l[l+1]:
                for i_lower in level_l_plus_1_to_l[l+1][i]:
                    z_i_level_l[l][i_lower] = z_i_level_l[l+1][i]

            for k in range(1,K+1):
                z_k_level_l[l][k] = []

                for i in z_k_level_l[l+1][k]:                       # for every upper (l+1) level unit i assigned to district k
                    for i_lower in level_l_plus_1_to_l[l+1][i]:     # for every lower (l) level unit i_lower uncoarsened from i
                        z_k_level_l[l][k].append(i_lower)           # assign i_lower to district k in lower level l

            """ Local search improvement at level l """
            z_k_improved, z_i_improved, obj_best = algorithm.local_search_flip(z_k_level_l[l], z_i_level_l[l], level_graph_l[l], 'sumdist', d_matrix_level_l[l])
            z_k_level_l[l], z_i_level_l[l] = z_k_improved.copy(), z_i_improved.copy()
            print("Compactness objective after level %i of uncoarsening:"%l,obj_best)

        return z_i_level_l[0], z_k_level_l[0]


class optimization:
    def optimize_MIP(coarsened_graph, d_matrix, warmstart_sol):
        print("Adding model")
        model = cplex.Cplex()
        nodes_list = list(coarsened_graph.nodes())

        edge_list = list(coarsened_graph.edges())
        pop_1 = {i:coarsened_graph.nodes[i]['p_rep'] for i in nodes_list}
        pop_2 = {i:coarsened_graph.nodes[i]['p_dem'] for i in nodes_list}
        pop = {i: coarsened_graph.nodes[i]['pop'] for i in nodes_list}

        M = (1+tau)*P_bar
        M_big = (1+tau)*P_bar

        var_counter = 0
        var_index_xij = {}
        for i in nodes_list:
            for j in nodes_list:
                var_index_xij[i,j] = var_counter
                var_counter += 1
        model.variables.add(names = ["x_%i,%i"%(i,j) for i in nodes_list for j in nodes_list], obj=[float(pop[j]*d_matrix[i][j]**2) for i in nodes_list for j in nodes_list], types=["B" for i in nodes_list for j in nodes_list])

        print("Added variables")
        start_constr_time = time.time()

        """ K assigments constraint """
        K_assignments = cplex.SparsePair(ind = ["x_%i,%i" %(i,i) for i in nodes_list], val = [1]*len(nodes_list))
        model.linear_constraints.add(lin_expr=[K_assignments], senses=["E"], rhs=[K])
        print("Added K assignment constraints")

        """ Every node is assigned to exactly one zone center constraint """
        for j in nodes_list:
            every_node_is_assigned = cplex.SparsePair(ind = ["x_%i,%i" %(i,j) for i in nodes_list], val = [1]*len(nodes_list))
            model.linear_constraints.add(lin_expr = [every_node_is_assigned], senses = ["E"], rhs = [1])
        print("Added exactly one district assignment constraints")

        """ A node is assigned to a zone center only if that zone center is indeed a zone center """
        for i in nodes_list:
            for j in [j for j in nodes_list if j!=i]:
                x_ij_x_ii = cplex.SparsePair(ind = ["x_%i,%i"%(i,j)]+["x_%i,%i"%(i,i)], val = [1]+[-1])
                model.linear_constraints.add(lin_expr = [x_ij_x_ii], senses = ["L"], rhs = [0])
        print("All assignment constraints added")
        #
        """ Population balance constraints constraints """
        for i in nodes_list:
            val_list1 = [pop[j] - (1-tau)*P_bar if j==i else pop[j] for j in nodes_list]
            val_list2 = [pop[j] - (1+tau)*P_bar if j==i else pop[j] for j in nodes_list]
            # print i, val_list1
            pop_balance_1 = cplex.SparsePair(ind = ["x_%i,%i"%(i,j) for j in nodes_list], val = val_list1)
            model.linear_constraints.add(lin_expr = [pop_balance_1], senses = ["G"], rhs = [0], names = ["pop_bal1_%i"%i])
            pop_balance_2 = cplex.SparsePair(ind = ["x_%i,%i"%(i,j) for j in nodes_list], val = val_list2)
            model.linear_constraints.add(lin_expr = [pop_balance_2], senses = ["L"], rhs = [0], names = ["pop_bal2_%i"%i])
        print("Population balance constraints added")

        """ Upper bound on efficiency gap defining constraints """
        if what_problem == 'EDP':
            model.variables.add(obj=[0]*len(nodes_list), lb=[0]*len(nodes_list), ub=[1]*len(nodes_list),types=["B"]*len(nodes_list), names = ["y_%i"%i for i in nodes_list])
            for i in nodes_list:
                model.variables.add(obj=[0], lb=[-float(M)/2], ub=[float(M)/2], types=["C"], names = ["w_%i"%(i)])
                for j in nodes_list:
                    model.variables.add(obj=[0], lb=[0], ub=[1], types=["B"], names = ["v_%i,%i"%(i,j)])

            print("Adding efficiency gap constraints")
            for i in nodes_list:
                majority_defining1 = cplex.SparsePair(ind = ["x_%i,%i"%(i,j) for j in nodes_list]+["y_%i"%i], val = [pop_1[j] - pop_2[j] for j in nodes_list]+[-M])
                model.linear_constraints.add(lin_expr = [majority_defining1], senses = ["L"], rhs = [0])
                majority_defining2 = cplex.SparsePair(ind = ["x_%i,%i"%(i,j) for j in nodes_list]+["y_%i"%i], val = [pop_1[j] - pop_2[j] for j in nodes_list]+[-M])
                model.linear_constraints.add(lin_expr = [majority_defining2], senses = ["G"], rhs = [-M])

            for i in nodes_list:
                y_i_x_ii = cplex.SparsePair(ind = ["y_%i"%(i)]+["x_%i,%i"%(i,i)], val = [1]+[-1])
                model.linear_constraints.add(lin_expr = [y_i_x_ii], senses = ["L"], rhs = [0])

            for i in nodes_list:
                for j in nodes_list:
                    vij_xij = cplex.SparsePair(ind = ["v_%i,%i"%(i,j)]+["x_%i,%i"%(i,j)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [vij_xij], senses = ["L"], rhs = [0])
                    vij_yi = cplex.SparsePair(ind = ["v_%i,%i"%(i,j)]+["y_%i"%(i)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [vij_yi], senses = ["L"], rhs = [0])
                    if i == j:
                        vij_yi_xij = cplex.SparsePair(ind = ["v_%i,%i"%(i,j)]+["y_%i"%(i)], val = [1]+[-1])
                    else:
                        vij_yi_xij = cplex.SparsePair(ind = ["v_%i,%i"%(i,j)]+["y_%i"%(i)]+["x_%i,%i"%(i,j)]+["x_%i,%i"%(i,i)], val = [1]+[-1]+[-1]+[1])
                    model.linear_constraints.add(lin_expr = [vij_yi_xij], senses = ["G"], rhs = [0])

            for i in nodes_list:
                net_wastedvotes_defining = cplex.SparsePair(ind = ["w_%i"%i]+["x_%i,%i"%(i,j) for j in nodes_list]+["v_%i,%i"%(i,j) for j in nodes_list], val = [1] + [-(3*pop_1[j]-pop_2[j])/2 for j in nodes_list] + [(pop_1[j] + pop_2[j]) for j in nodes_list])
                model.linear_constraints.add(lin_expr = [net_wastedvotes_defining], senses = ["E"], rhs = [0])

            netwastedvotes_ub_enforcing = cplex.SparsePair(ind = ["w_%i"%i for i in nodes_list], val = [1]*len(nodes_list))
            model.linear_constraints.add(lin_expr = [netwastedvotes_ub_enforcing], senses = ["L"], rhs = [epsilon_EG*(sum(pop_1[j]+pop_2[j] for j in nodes_list))])
            model.linear_constraints.add(lin_expr = [netwastedvotes_ub_enforcing], senses = ["G"], rhs = [-epsilon_EG*(sum(pop_1[j]+pop_2[j] for j in nodes_list))])

        """ Competitiveness constraints (difference in highest and lowest) """
        if what_problem == 'CmpttvDP':
            print("Competitiveness constraints (difference in highest and lowest) adding")
            model.variables.add(obj=[0], types=["C"], names = ["diff"])
            for i in nodes_list:
                for j in nodes_list:
                    model.variables.add(obj=[0], lb = [0], ub = [1], types=["C"], names = ["z_%i,%i"%(i,j)])
            for i in nodes_list:
                diff_defining1 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j) for j in nodes_list]+["x_%i,%i"%(i,j) for j in nodes_list], val = [(pop_1[j] + pop_2[j]) for j in nodes_list]+[(pop_1[j] - pop_2[j]) for j in nodes_list])
                model.linear_constraints.add(lin_expr = [diff_defining1], senses = ["G"], rhs = [0])
                diff_defining2 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j) for j in nodes_list]+["x_%i,%i"%(i,j) for j in nodes_list], val = [(pop_1[j] + pop_2[j]) for j in nodes_list]+[-(pop_1[j] - pop_2[j]) for j in nodes_list])
                model.linear_constraints.add(lin_expr = [diff_defining2], senses = ["G"], rhs = [0])
            for i in nodes_list:
                for j in nodes_list:
                    z_ij_defining1 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j)]+["x_%i,%i"%(i,j)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [z_ij_defining1], senses = ["L"], rhs = [0])
                    z_ij_defining2 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j)]+["diff"], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [z_ij_defining2], senses = ["L"], rhs = [0])
                    z_ij_defining3 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j)]+["diff"]+["x_%i,%i"%(i,j)], val = [1]+[-1]+[-1])
                    model.linear_constraints.add(lin_expr = [z_ij_defining3], senses = ["G"], rhs = [-1])
                    z_ij_defining4 = cplex.SparsePair(ind = ["z_%i,%i"%(i,j)], val = [1])
                    model.linear_constraints.add(lin_expr = [z_ij_defining4], senses = ["G"], rhs = [0])

            diff_upperbound = cplex.SparsePair(ind = ["diff"], val = [1])
            model.linear_constraints.add(lin_expr = [diff_upperbound], senses = ["L"], rhs = [epsilon_cmpttv*2])

        """ Partisa Asymmetry constraints """
        if what_problem == 'SDP':
            print("Adding partisan assymmetry constraints")

            model.variables.add(obj=[0]*len(nodes_list), lb=[0]*len(nodes_list), ub=[1]*len(nodes_list), types=["C"]*len(nodes_list), names = ["v_%i"%(i) for i in nodes_list])
            model.variables.add(obj=[0]*K, lb=[0]*K, ub=[1]*K, types=["C"]*K, names = ["alpha_%i"%(k) for k in range(1,K+1)])
            model.variables.add(obj=[0]*K, lb=[0]*K, ub=[1]*K, types=["C"]*K, names = ["s_%i"%(k) for k in range(1,K+1)])
            model.variables.add(obj=[0]*K, lb=[0]*K, ub=[1]*K, types=["C"]*K, names = ["obj_PA_%i"%(k) for k in range(1,K+1)])
            model.variables.add(obj=[1], lb=[0], ub=[1], types=["C"], names = ["obj_PA"])
            for j in nodes_list:
                model.variables.add(obj=[0]*len(nodes_list), lb=[0]*len(nodes_list), ub=[1]*len(nodes_list), types=["C"]*len(nodes_list), names = ["beta_%i,%i"%(i,j) for i in nodes_list])
            for k in range(1,K+1):
                model.variables.add(obj=[0]*len(nodes_list), lb=[0]*len(nodes_list), ub=[1]*len(nodes_list), types=["B"]*len(nodes_list), names = ["delta_%i,%i"%(i,k) for i in nodes_list])
                model.variables.add(obj=[0]*len(nodes_list), lb=[0]*len(nodes_list), ub=[1]*len(nodes_list), types=["B"]*len(nodes_list), names = ["gamma_%i,%i"%(i,k) for i in nodes_list])
                model.variables.add(obj=[0]*K, lb=[0]*K, ub=[100]*K, types=["C"]*K, names = ["mu_%i,%i"%(k,m) for m in range(1,K+1)])
                model.variables.add(obj=[0]*K, lb=[0]*K, ub=[1]*K, types=["B"]*K, names = ["omega_%i,%i"%(k,m) for m in range(1,K+1)])
                model.variables.add(obj=[0]*K, lb=[0]*K, ub=[1]*K, types=["B"]*K, names = ["omegadash_%i,%i"%(k,m) for m in range(1,K+1)])

            print("defined partisan assymmetry variables")

            for i in nodes_list:
                beta_defining1 = cplex.SparsePair(ind = ["beta_%i,%i"%(i,j) for j in nodes_list]+["x_%i,%i"%(i,j) for j in nodes_list], val = [float(pop_1[j]+pop_2[j]) for j in nodes_list]+[-float(pop_1[j]) for j in nodes_list])
                model.linear_constraints.add(lin_expr = [beta_defining1], senses = ["E"], rhs = [0])

                v_x_relationship = cplex.SparsePair(ind = ["v_%i"%(i)]+["x_%i,%i"%(i,i)], val = [1]+[-1])
                model.linear_constraints.add(lin_expr = [v_x_relationship], senses = ["L"], rhs = [0])

                for j in nodes_list:
                    beta_defining2 = cplex.SparsePair(ind = ["beta_%i,%i"%(i,j)]+["x_%i,%i"%(i,j)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [beta_defining2], senses = ["L"], rhs = [0])

                    beta_v_relationship1 = cplex.SparsePair(ind = ["beta_%i,%i"%(i,j)]+["v_%i"%(i)]+["x_%i,%i"%(i,j)], val = [1]+[-1]+[-1])
                    model.linear_constraints.add(lin_expr = [beta_v_relationship1], senses = ["G"], rhs = [-1])

                    beta_v_relationship2 = cplex.SparsePair(ind = ["beta_%i,%i"%(i,j)]+["v_%i"%(i)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [beta_v_relationship2], senses = ["L"], rhs = [0])

                for k in range(1,K+1):
                    alpha_defining1 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["v_%i"%(i)]+["gamma_%i,%i"%(i,k)], val = [1]+[-1]+[1])
                    model.linear_constraints.add(lin_expr = [alpha_defining1], senses = ["G"], rhs = [0])

                    alpha_defining2 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["v_%i"%(i)]+["delta_%i,%i"%(i,k)], val = [1]+[-1]+[1])
                    model.linear_constraints.add(lin_expr = [alpha_defining2], senses = ["L"], rhs = [1])

            for k in range(1,K+1):
                gamma_defining = cplex.SparsePair(ind = ["gamma_%i,%i"%(i,k) for i in nodes_list], val = [1 for i in nodes_list])
                model.linear_constraints.add(lin_expr = [gamma_defining], senses = ["E"], rhs = [k-1])

                delta_defining = cplex.SparsePair(ind = ["delta_%i,%i"%(i,k) for i in nodes_list], val = [1 for i in nodes_list])
                model.linear_constraints.add(lin_expr = [delta_defining], senses = ["E"], rhs = [k])

            for k in range(1,K+1):
                for m in range(1,K+1):
                    mu_omegas_relationship1 = cplex.SparsePair(ind = ["mu_%i,%i"%(k,m)]+["omega_%i,%i"%(k,m)], val = [1]+[-1])
                    model.linear_constraints.add(lin_expr = [mu_omegas_relationship1], senses = ["L"], rhs = [0])

                    mu_omegas_relationship2 = cplex.SparsePair(ind = ["mu_%i,%i"%(k,m)]+["omegadash_%i,%i"%(k,m)], val = [1]+[1])
                    model.linear_constraints.add(lin_expr = [mu_omegas_relationship2], senses = ["G"], rhs = [1])

            for k in range(1,K+1):
                for m in range(1,K+1):
                    if k!=m:
                        alpha_mu_omega_relationship1 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["alpha_%i"%(m)]+["omega_%i,%i"%(k,m)], val = [-1]+[1]+[-1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omega_relationship1], senses = ["L"], rhs = [-0.5])
                        alpha_mu_omega_relationship2 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["alpha_%i"%(m)]+["mu_%i,%i"%(k,m)]+["omega_%i,%i"%(k,m)], val = [-1]+[1]+[-1]+[-1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omega_relationship2], senses = ["G"], rhs = [-2])
                        alpha_mu_omegadash_relationship1 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["alpha_%i"%(m)]+["omegadash_%i,%i"%(k,m)], val = [-1]+[1]+[1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omegadash_relationship1], senses = ["G"], rhs = [0.5])
                        alpha_mu_omegadash_relationship2 = cplex.SparsePair(ind = ["alpha_%i"%(k)]+["alpha_%i"%(m)]+["mu_%i,%i"%(k,m)]+["omegadash_%i,%i"%(k,m)], val = [-1]+[1]+[-1]+[1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omegadash_relationship2], senses = ["L"], rhs = [1])
                    else:
                        alpha_mu_omega_relationship1 = cplex.SparsePair(ind = ["omega_%i,%i"%(k,m)], val = [-1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omega_relationship1], senses = ["L"], rhs = [-0.5])
                        alpha_mu_omega_relationship2 = cplex.SparsePair(ind = ["mu_%i,%i"%(k,m)]+["omega_%i,%i"%(k,m)], val = [-1]+[-1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omega_relationship2], senses = ["G"], rhs = [-2])
                        alpha_mu_omegadash_relationship1 = cplex.SparsePair(ind = ["omegadash_%i,%i"%(k,m)], val = [1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omegadash_relationship1], senses = ["G"], rhs = [0.5])
                        alpha_mu_omegadash_relationship2 = cplex.SparsePair(ind = ["mu_%i,%i"%(k,m)]+["omegadash_%i,%i"%(k,m)], val = [-1]+[1.5])
                        model.linear_constraints.add(lin_expr = [alpha_mu_omegadash_relationship2], senses = ["L"], rhs = [1])

                s_defining = cplex.SparsePair(ind = ["s_%i"%k]+["mu_%i,%i"%(k,m) for m in range(1,K+1)], val = [K]+[-1]*K)
                model.linear_constraints.add(lin_expr = [s_defining], senses = ["E"], rhs = [0])

                OBJ_PA_k_defining1 = cplex.SparsePair(ind = ["obj_PA_%i"%k]+["s_%i"%(k)]+["s_%i"%(K-k+1)], val = [1]+[-1]+[-1])
                model.linear_constraints.add(lin_expr = [OBJ_PA_k_defining1], senses = ["G"], rhs = [-1])
                OBJ_PA_k_defining2 = cplex.SparsePair(ind = ["obj_PA_%i"%k]+["s_%i"%(k)]+["s_%i"%(K-k+1)], val = [1]+[1]+[1])
                model.linear_constraints.add(lin_expr = [OBJ_PA_k_defining2], senses = ["G"], rhs = [1])

            obj_PA_defining = cplex.SparsePair(ind = ["obj_PA"]+["obj_PA_%i"%(k) for k in range(1,K+1)], val = [K]+[-1 for k in range(1,K+1)])
            model.linear_constraints.add(lin_expr = [obj_PA_defining], senses = ["E"], rhs = [0])

            upperbound_on_PA = cplex.SparsePair(ind = ["obj_PA"], val = [1])
            model.linear_constraints.add(lin_expr = [upperbound_on_PA], senses = ["L"], rhs = [epsilon_PA])

        print("Adding flow-based contiguity constraints")
        start_time_flowconts = time.time()
        """ Contiguity using flow constraints """
        for i in coarsened_graph.nodes():
            model.variables.add(obj=[0]*len(edge_list), lb=[0] * len(edge_list), types=["C"] * len(edge_list), names = ["f_%i,%i,%i"%(i,j,v) for (j,v) in edge_list])
            model.variables.add(obj=[0]*len(edge_list), lb=[0] * len(edge_list), types=["C"] * len(edge_list), names = ["f_%i,%i,%i"%(i,v,j) for (j,v) in edge_list])

        for i in coarsened_graph.nodes():
            model.variables.add(obj=[0]*len(edge_list), lb=[0] * len(edge_list),types=["C"] * len(edge_list), names = ["f_%i,%i,%i"%(i,j,v) for (j,v) in edge_list])
            model.variables.add(obj=[0]*len(edge_list), lb=[0] * len(edge_list),types=["C"] * len(edge_list), names = ["f_%i,%i,%i"%(i,v,j) for (j,v) in edge_list])
        for i in nodes_list:
            for j in nodes_list:
                if j!=i:
                    flow_contiguity_1 = cplex.SparsePair(ind = ["f_%i,%i,%i"%(i,j,v) for v in coarsened_graph.neighbors(j)]+["f_%i,%i,%i"%(i,v,j) for v in coarsened_graph.neighbors(j)]+["x_%i,%i"%(i,j)],
                                    val = [1 for v in coarsened_graph.neighbors(j)]+[-1 for v in coarsened_graph.neighbors(j)]+[1])
                    model.linear_constraints.add(lin_expr = [flow_contiguity_1], senses = ["E"], rhs = [0])
                else:
                    flow_contiguity_1 = cplex.SparsePair(ind = ["f_%i,%i,%i"%(i,i,v) for v in coarsened_graph.neighbors(i)]+["f_%i,%i,%i"%(i,v,i) for v in coarsened_graph.neighbors(i)]+["x_%i,%i"%(i,v) for v in nodes_list if v!=i],
                                    val = [1 for v in coarsened_graph.neighbors(i)]+[-1 for v in coarsened_graph.neighbors(i)]+[-1 for v in nodes_list if v!=i])
                    model.linear_constraints.add(lin_expr = [flow_contiguity_1], senses = ["E"], rhs = [0])

                flow_contiguity_2 = cplex.SparsePair(ind = ["f_%i,%i,%i"%(i,j,v) for v in coarsened_graph.neighbors(j)]+["x_%i,%i"%(i,j)],
                                val = [1 for v in coarsened_graph.neighbors(j)]+[-(len(nodes_list))])
                model.linear_constraints.add(lin_expr = [flow_contiguity_2], senses = ["L"], rhs = [0])

        model.objective.set_sense(model.objective.sense.minimize)
        print("Added all the constraints at time",time.time() - start_constr_time, "secs and now solving")

        try:
            # model.set_log_stream(None)
            # model.set_error_stream(None)
            # model.set_warning_stream(None)
            # model.set_results_stream(None)

            model.parameters.timelimit.set(mip_timelimit)
            model.parameters.mip.tolerances.mipgap.set(0)
            if warmstart_sol != {}:
                warmstart = cplex.SparsePair(ind = [key for key,val in list(warmstart_sol.items())], val = [val for key,val in list(warmstart_sol.items())])
                model.MIP_starts.add(warmstart, model.MIP_starts.effort_level.solve_MIP, "initial start")

            start_time_sol = time.time()
            start_time_ticks = model.get_dettime()
            model.solve()

        except CplexSolverError as e:
            print("Exception raised during solve: " + e)
        else:
            solution = model.solution
            print("The status is",solution.status[solution.get_status()])
            print("Time elapsed:", time.time() - start_time_sol)
            sol_time = time.time() - start_time_sol
            sol_time_ticks = model.get_dettime() - start_time_ticks
            if solution.status[solution.get_status()] == 'MIP_infeasible' or solution.status[solution.get_status()] == 'MIP_time_limit_infeasible':
                return {}, {}, 0, sol_time, sol_time_ticks, solution.progress.get_num_nodes_processed(), 0, 1
            else:
                print("Objective value:", solution.get_objective_value())

                z_copy, z_i, z_k = {}, {}, {k: set([]) for k in range(1,K+1)}
                k_curr = 1
                for i in nodes_list:
                    if abs(solution.get_values("x_%i,%i"%(i,i)) - 1) < 0.01:
                        z_i[i] = k_curr
                        z_copy[i] = k_curr
                        z_k[k_curr].add(i)
                        k_curr += 1

                for j in nodes_list:
                    for i in z_copy:
                        if solution.get_values("x_%i,%i"%(i,j)):
                            z_i[j] = z_copy[i]
                            if i!=j:
                                z_k[z_copy[i]].add(j)

                z_k = {k: list(z_k[k]) for k in z_k}

                return z_i, z_k, solution.get_objective_value(), sol_time, sol_time_ticks, solution.progress.get_num_nodes_processed(), solution.MIP.get_mip_relative_gap(), 0


    def solve_instance(coarsened_graph, what_problem, d_matrix_level_l, level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph_l):
        """ Solves the coarsened instance using the epsilon-constraint method, uncoarsens a Pareto-optimal solution, and stores the solution into a file """

        global epsilon_EG, epsilon_cmpttv, epsilon_PA

        """ Initiate the initial value of epsilon """
        if what_problem == 'EDP':
            epsilon_EG = 0.5
        elif what_problem == 'SDP':
            epsilon_PA = 1
        elif what_problem == 'CmpttvDP':
            epsilon_cmpttv = 1

        solution_index = 1 # Solution counter epsilon-constraint method
        while(1):
            """ Generate an initial feasible solution """
            z_k_initial, z_i_initial = {}, {}
            start_time_initial_heuristic = time.time()
            if initial_heuristic_or_not:
                compactness_obj_best = 9**20
                for iter in range(number_of_iterations_multistart):
                    z_i, z_k = algorithm.vickrey_initial_solution_heuristic(coarsened_graph, d_matrix_level_l[L], start_time_initial_heuristic)
                    z_k, z_i, compactness_obj = algorithm.local_search_flip(z_k, z_i, coarsened_graph, 'sumdist', d_matrix_level_l[L])
                    if compactness_obj < compactness_obj_best:
                        compactness_obj_best = compactness_obj
                        z_i_initial = z_i.copy()
                        z_k_initial = z_k.copy()
                        print("new best compactness obj:",compactness_obj_best)
                    if time.time() - start_time_initial_heuristic > initialheuristic_timelimit:
                        break

            """ Solve MIP """
            warmstart_sol, warmstart_dict = algorithm.get_warmsart_from_z(z_i_initial, z_k_initial, coarsened_graph, d_matrix_level_l[L])
            z_i_opt, z_k_opt, opt_obj, sol_time, sol_time_ticks, bnb_nodes, optimality_gap, MIP_infeasible  = optimization.optimize_MIP(coarsened_graph, d_matrix_level_l[L], warmstart_sol)

            # compute_metrics.plot_plan(coarsened_graph, z_k_opt, z_i_opt)

            if MIP_infeasible:
                break
            else:
                """ Update epsilon values """
                if what_problem == 'EDP':
                    opt_effgap = compute_metrics.evaluate_effgap(z_k_opt, coarsened_graph)
                    epsilon_EG = opt_effgap - 10**-7
                elif what_problem == 'SDP':
                    opt_PA = compute_metrics.compute_partisan_symmetry_with_formula(z_k_opt, coarsened_graph)
                    epsilon_PA = opt_PA - 10**-7
                elif what_problem == 'CmpttvDP':
                    opt_cmpttv = compute_metrics.evaluate_max_margin(z_k_opt, coarsened_graph)
                    epsilon_cmpttv = opt_cmpttv - 10**-7

                """ Uncoarsen instance """
                z_i_final, z_k_final = algorithm.uncoarsen_with_local_search(z_i_opt, z_k_opt, level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph_l, d_matrix_level_l)

                """ Compute fairness metrics for the plan """
                compactness = compute_metrics.evaluate_compactness_sumdist(z_k_final, level_graph_l[0], d_matrix_level_l[0])[0]
                EG = compute_metrics.evaluate_effgap(z_k_final, level_graph_l[0])
                PA = compute_metrics.compute_partisan_symmetry_with_formula(z_k_final, level_graph_l[0])
                max_margin =  compute_metrics.evaluate_max_margin(z_k_final, level_graph_l[0])

                """ Record the approximate Pareto-optimal solutions to .csv file """
                def write_plan_to_file(z_i, compactness, EG, PA, max_margin, opt_gap):
                    if what_problem == 'CompactDP':
                        filename = 'Results/Most_fair_maps/mostcompactmap_'+str(tau)+'_'+str(matching_strategy)+'_compactness'+str(compactness)+'.csv'
                    else:
                        filename = 'Results/Approx_Pareto_optimal_maps/'+what_problem
                        filename += '/map_'+matching_strategy+'_'+str(solution_index)
                        if what_problem == 'EDP':
                            filename += '_effgap'+str(abs(EG))
                        elif what_problem == 'CmpttvDP':
                            filename += '_cmpttv'+str(max_margin)
                        elif what_problem == 'SDP':
                            filename += '_PA'+str(PA)

                        filename += '_optgap'+str(optimality_gap)+'.csv'

                    with open(filename, 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['GEOID', 'district'])
                        for key, value in list(z_i.items()):
                            writer.writerow([key, value])

                write_plan_to_file(z_i_final, compactness, EG, PA, max_margin, optimality_gap)
                solution_index += 1

                if what_problem == 'CompactDP':
                    return

        return


def main():
    """ Set problem instance parameters """
    global K, tau, P_bar, what_problem
    K = 8 # number of districts
    tau = 0.02 # population balance deviation threshold
    what_problem = 'CompactDP' # what problem is solved: 'EDP','SDP','CmpttvDP','CompactDP'

    """ Set algorithm parameters """
    global L, matching_strategy, initial_heuristic_or_not, number_of_iterations_multistart, initialheuristic_timelimit, mip_timelimit, P_bar
    L = 3 # number of levels of coarsening
    matching_strategy = 'ML' # what matching is used for coarsening: 'ML' for maximal, 'MM' for maximum
    mip_timelimit = 86400 # time limit for MIP; 21600 for 6 hours, 86400 for 24 hours

    initial_heuristic_or_not = 1 # 1 if the multistart initial solution heuristic should be used; 0 if not
    number_of_iterations_multistart = 100 # number of iterations of the multistart initial solution heuristic
    initialheuristic_timelimit = 3600 # time limit for running the initial solution heuristic

    """ Read input data """
    G_0, d_matrix = input.read_data_tract()
    P_bar = int(sum(G_0.nodes[i]['pop'] for i in G_0.nodes())/K) # ideal distict population

    """ Coarsen instance """
    coarsened_graph, coarsened_graph_before_aggregation, level_l_to_l_plus_1, level_l_plus_1_to_l, d_matrix_level_l, level_graph_l = coarsen.coarsen_multilevel(G_0)

    """ Solve the coarsest instance, uncoarsen, and store the district plan in a file """
    optimization.solve_instance(coarsened_graph, what_problem, d_matrix_level_l, level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph_l)


if __name__ == "__main__":
    main()

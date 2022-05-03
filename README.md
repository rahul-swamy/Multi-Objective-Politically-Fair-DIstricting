# Code companion for Multi-objective Optimization for Politically Fair Districting: A Scalable Multilevel Approach

This repository contains the code, data and results pertaining to the manuscript “_Multi-objective Optimization for Politically Fair Districting: A Scalable Multilevel Approach_” by Rahul Swamy, Douglas M. King, and Sheldon H. Jacobson. This work is accepted at _Operations Research_.

The **Datasets** folder contains the instance of Wisconsin. the root folder contains the code, and the **Results** folder contains the results such as the 3 levels of coarsened graphs, the approximate-Pareto optimal solutions for the 3 bi-objective problems solved in the paper and the most fair maps from among them.

The **Datasets** folder contains:
- **WI_adjacency_tracts.csv**: The adjacency list for the census tracts of Wisconsin from Census 2010.
- **WI_2010census_population_tracts.csv**: The population in the census tracts of Wisconsin from Census 2010.
- **WI_2012presidential_tracts.csv**: The 2012 presidential election results at the census tract level from McGovern (2017).
- **WI_2016presidential_tracts.csv**: The 2016 presidential election results at the census tract level from The Guardian (2018).
- **WI_coordinates_tracts.csv**: The geo-coordinates of the census tracts from Census 2010.


The following are the codes contained in the root folder.
- **generate_plans.py**: This code is the main function to generate district plans. This reads the Wisconsin instance from the **Datasets** folder, implements the multilevel algorithm + epsilon-constraint method to solve a chosen bi-objective problem, and stores the approximate Pareto-optimal solutions to the **Results** folder.
- **compute_metrics.py**: This code contains functions that calculate the fairness metrics and other quantities related to the districting problem. This code also visually plots a given district plan. The main function takes a filename as input for a district plan, prints its fairness metrics, and visually plots the district plan.
- **maxcardinalitymatching.py**: This code contains the Edmonds' algorithm to find a maximum cardinality matching implemented by Eppstein (2003).

The **Results** folder contains:
- **Most_fair_maps**: This folder has the most fair district plans with respect to the efficiency gap (**most_equitable_map.csv**), partisan asymmetry (**most_symmetric_map.csv**), and competitiveness (**most_cmpttv_map.csv**). It also has the most compact maps for varying population thresholds tau in [2%, 1%, .5%, .25%].
- **Approx_Pareto_optimal_maps**: This folder has the approximate Pareto-optimal plans for the 3 bi-objective problems. Each plan's file name contains the matching strategy used, an index of the plan, its fairness value corresponding to the objective used to obtain it, and the optimality gap return by CPLEX at the coarsest level. 
- **Coarsened_graphs**: This folder contains plots of 3 coarse graph levels when coarsening using ML.

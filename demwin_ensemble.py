#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import time
import random
import numpy as np
import pandas as pd

start_time = time.time()

random.seed(382946)

# Loads the graph from the shapefile, representing Illinois's districts
il_graph = Graph.from_file("./IL/IL.shp")

# Prints the attributes of the first node (for debugging)
print(il_graph.nodes[0])
#%%


# Initializes a partition of the graph with the updaters for the electoral data
initial_partition = Partition(
    il_graph,
    assignment="SSD",
    updaters={
        "population": Tally("TOTPOP", alias="population"),
        "cut_edges": cut_edges,
        "dem_pres_votes": Tally("G20PRED", alias="dem_pres_votes"),
        "rep_pres_votes": Tally("G20PRER", alias="rep_pres_votes"),
        "dem_sen_votes": Tally("G20USSD", alias="dem_sen_votes"),
        "rep_sen_votes": Tally("G20USSR", alias="rep_sen_votes")
    }
)
# Calculates the total population and ideal population per district
tot_pop = sum([il_graph.nodes()[v]['TOTPOP'] for v in il_graph.nodes()])
num_dist = 59 # Number of districts
ideal_pop = tot_pop/num_dist # Ideal population per district
pop_tolerance = .1  # Allowed deviation from ideal population

# Defines the parameters for the random walk proposal used in redistricting
rw_proposal = partial(recom, ## how you choose a next districting plan
                      pop_col = "TOTPOP", ## What data describes population? 
                      pop_target = ideal_pop, ## What the target/ideal population is for each district 
                                              ## (we calculated ideal pop above)
                      epsilon = pop_tolerance,  ## how far from ideal population you can deviate
                                              ## (we set pop_tolerance above)
                      node_repeats = 1 ## number of times to repeat bipartition.  Can increase if you get a BipartitionWarning
                      )


population_constraint = constraints.within_percent_of_ideal_population(
    initial_partition, 
    pop_tolerance, 
    pop_key="population")

#sets up the markov chain
our_random_walk = MarkovChain(
    proposal = rw_proposal, 
    constraints = [population_constraint],
    accept = always_accept, # Accept every proposed plan that meets the population constraints
    initial_state = initial_partition, 
    #20000 times made two graphs with 2 different seeds that looked the same
    total_steps = 50000) 

#ensembles keeping track of cut edges, number of districts that are majority latino, and num of districts that dems won
cutedge_ensemble = []
pres_demwin_ensemble = []
sen_demwin_ensemble = []
mean_median_diff_pres = []
efficiency_gap_pres = []
mean_median_diff_sen = []
efficiency_gap_sen = []


def mean_median(partition, dem_key, rep_key):
    # Counts the democratic and republican votes for each district
    dem_votes = [partition[dem_key][i + 1] for i in range(num_dist)]
    rep_votes = [partition[rep_key][i + 1] for i in range(num_dist)]
    
    # Calculates the vote share for the Democratic party where total votes are non-zero
    vote_shares = [d / (d + r) for d, r in zip(dem_votes, rep_votes) if d + r > 0]
    
    # Calculates the median and mean of these Democratic vote shares
    median = np.median(vote_shares)
    mean = np.mean(vote_shares)
    
    # Returns the difference between the median and the mean vote share
    return median - mean


def efficiency_gap(partition, dem_key, rep_key):
    wasted_d = 0  # Initializes the wasted votes for Democrats
    wasted_r = 0  # Initializes the wasted votes for Republicans
    
    # Loops through each district to calculate wasted votes
    for i in range(num_dist):
        d_votes = partition[dem_key][i + 1]
        r_votes = partition[rep_key][i + 1]
        total_votes = d_votes + r_votes
        
        # Calculate wasted votes depending on which party won the district
        if d_votes > r_votes:
            wasted_d += d_votes - (total_votes / 2)  # Democrats win, calculates their wasted votes
            wasted_r += r_votes  # All Republican votes are wasted
        else:
            wasted_d += d_votes  # All Democrat votes are wasted
            wasted_r += r_votes - (total_votes / 2)  # Republicans win, calculates their wasted votes
            
    # Calculates and return the efficiency gap, which is normalized by the total votes
    return (wasted_r - wasted_d) / sum(list(partition[dem_key].values()) + list(partition[rep_key].values()))


# Iterates over the partitions generated in the Markov chain
for part in our_random_walk:
    cutedge_ensemble.append(len(part["cut_edges"]))
    # Counts the number of districts won by Democrats in presidential and senatorial elections
    num_dem_win_sen = 0
    num_dem_win_pres = 0
    for i in range(num_dist):
        if(part["dem_pres_votes"][i+1] > part["rep_pres_votes"][i+1]):
            num_dem_win_pres  += 1
    for i in range(num_dist):
        if(part["dem_sen_votes"][i+1] > part["rep_sen_votes"][i+1]):
            num_dem_win_sen += 1
    # Appends the counts to respective lists
    sen_demwin_ensemble.append(num_dem_win_sen)
    pres_demwin_ensemble.append(num_dem_win_pres)
    # Calculates and tracks the mean-median difference and efficiency gap for both elections
    mm_diff_pres = mean_median(part, "dem_pres_votes", "rep_pres_votes")
    eg_pres = efficiency_gap(part, "dem_pres_votes", "rep_pres_votes")
    mm_diff_sen = mean_median(part, "dem_sen_votes", "rep_sen_votes")
    eg_sen = efficiency_gap(part, "dem_sen_votes", "rep_sen_votes")
    
    mean_median_diff_pres.append(mm_diff_pres)
    efficiency_gap_pres.append(eg_pres)
    mean_median_diff_sen.append(mm_diff_sen)
    efficiency_gap_sen.append(eg_sen)

# Plotting the histogram of the cut edges from the ensemble analysis
plt.figure()
plt.hist(cutedge_ensemble, align='left')
plt.title('Cut Edges')
plt.show()

# Plotting the histogram of the Presidential elections won by Democrats
plt.figure()
plt.hist(pres_demwin_ensemble, align='left')
plt.title('Presidential Elections Won by Democrats')
plt.show()

# Plotting the histogram of the Senate elections won by Democrats
plt.figure()
plt.hist(sen_demwin_ensemble, align='left')
plt.title('Senate Elections Won by Democrats')
plt.show()

# Plotting the histogram for the mean-median difference in Presidential elections
plt.figure()
plt.hist(mean_median_diff_pres, align='left')
plt.title("Mean-Median Difference for Presidential Election")
plt.show()

# Plotting the histogram for the efficiency gap in Presidential elections
plt.figure()
plt.hist(efficiency_gap_pres, align='left')
plt.title("Efficiency Gap for Presidential Election")
plt.show()

# Plotting the histogram for the mean-median difference in Senate elections
plt.figure()
plt.hist(mean_median_diff_sen, align='left')
plt.title("Mean-Median Difference for Senate Election")
plt.show()

# Plotting the histogram for the efficiency gap in Senate elections
plt.figure()
plt.hist(efficiency_gap_sen, align='left')
plt.title("Efficiency Gap for Senate Election")
plt.show()

# Creates a dictionary with ensemble analysis results
col_names = {'cutedge_ensemble': cutedge_ensemble, 
             'pres_demwin_ensemble': pres_demwin_ensemble,
             'sen_demwin_ensemble': sen_demwin_ensemble,
             'mean_median_diff_pres': mean_median_diff_pres,
             'mean_median_diff_sen':mean_median_diff_sen,
             'efficiency_gap_pres': mean_median_diff_pres,
             'efficiency_gap_sen': efficiency_gap_sen}

# Converts the dictionary to a DataFrame
ensembles = pd.DataFrame(col_names)

# Saves the DataFrame to a CSV file
ensembles.to_csv('ensembles.csv')

end_time = time.time()
print("The time of execution of above program is :",
      (end_time-start_time)/60, "mins")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:26:42 2024

@author: eveomett

Author: Ellen Veomett

for AI for Redistricting

Lab 3, spring 2024
"""


import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import time
import random


start_time = time.time()

random.seed(382946)

il_graph = Graph.from_file("./IL/IL.shp")

print(il_graph.nodes[0])
#%%


"""
DONE: Lab 3!  See all the instructions on Canvas
"""

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

tot_pop = sum([il_graph.nodes()[v]['TOTPOP'] for v in il_graph.nodes()])
num_dist = 59
ideal_pop = tot_pop/num_dist
pop_tolerance = .1

#random walk parameters
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
    total_steps = 50) 

#ensembles keeping track of cut edges, number of districts that are majority latino, and num of districts that dems won
cutedge_ensemble = []
pres_demwin_ensemble = []
sen_demwin_ensemble = []

#runs the markov chain
for part in our_random_walk:
    cutedge_ensemble.append(len(part["cut_edges"]))
    
    num_dem_win = 0
    for i in range(num_dist):
        if(part["dem_pres_votes"][i+1] > part["rep_pres_votes"][i+1]):
            num_dem_win += 1
    pres_demwin_ensemble.append(num_dem_win)
    
    num_dem_win = 0
    for i in range(num_dist):
        if(part["dem_sen_votes"][i+1] > part["rep_sen_votes"][i+1]):
            num_dem_win += 1
    sen_demwin_ensemble.append(num_dem_win)

#draws the histograms
plt.figure()
plt.hist(cutedge_ensemble, align = 'left')
plt.show()

plt.figure()
plt.hist(pres_demwin_ensemble, align='left')
plt.show()

plt.figure()
plt.hist(sen_demwin_ensemble, align='left')
plt.show()
    
print('CUTEDGE:')
print(cutedge_ensemble)
print('PRESDEMWIN:')
print(pres_demwin_ensemble)
print("SENDEMWIN:")
print(sen_demwin_ensemble)

#takes like 100-110 minutes
end_time = time.time()
print("The time of execution of above program is :",
      (end_time-start_time)/60, "mins")
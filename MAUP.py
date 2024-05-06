#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import maup
import time
from pyproj import CRS
import pickle
from gerrychain import Graph

maup.progress.enabled = True

#population
start_time = time.time()
population_df = gpd.read_file("./il_pl2020_b/il_pl2020_p2_b.shp")
end_time = time.time()
print("The time to import the Population shape file is:",
      (end_time-start_time)/60, "mins")

#voting age population
start_time = time.time()
vap_df= gpd.read_file("./il_pl2020_b/il_pl2020_p4_b.shp")
end_time = time.time()
print("The time to import VAP shape file is:",
      (end_time-start_time)/60, "mins")

#2020 election
start_time = time.time()
election_df = gpd.read_file("./il_vest_20/il_vest_20.shp")
end_time = time.time()
print("The time to import the 2020 VEST shape file is:",
      (end_time-start_time)/60, "mins")

#congressional district data
start_time = time.time()
district_df = gpd.read_file("./il_sldu_2021/il_sldu_2021.shp")
end_time = time.time()
print(district_df.geometry)
print("The time to import the Congressional District data shapefile is:",
      (end_time-start_time)/60, "mins")

# Print column names from imported DataFrames 
print("population:")
print(population_df.columns)
print("VAP:")
print(vap_df.columns)
print(election_df.columns)
print("district:")
print(district_df.columns)
print("election:")
print(election_df)

# Checks data integrity using Maup doctor
print(maup.doctor(population_df)) # - True
print(maup.doctor(vap_df)) # - True
print(maup.doctor(election_df)) # - True
print(maup.doctor(district_df)) # - True

# Convert DataFrames to their estimated UTM coordinate reference systems
election_df = election_df.to_crs(election_df.estimate_utm_crs())
population_df = population_df.to_crs(population_df.estimate_utm_crs())
vap_df = vap_df.to_crs(vap_df.estimate_utm_crs())
district_df = district_df.to_crs(district_df.estimate_utm_crs())
# Prints current CRS of DataFrames
print(population_df.crs,"    ", vap_df.crs, "      ", election_df.crs, "     ", district_df.crs)

# Begins precincts assignment from block data
print("start blocks to precincts assignment")
blocks_to_precincts_assignment = maup.assign(population_df.geometry, election_df.geometry)
vap_blocks_to_precincts_assignment = maup.assign(vap_df.geometry, election_df.geometry)

#Gets only these columns from pop and vap shapefiles
pop_column_names = ['P0020001', 'P0020002', 'P0020005', 'P0020006', 'P0020007',
                    'P0020008', 'P0020009', 'P0020010', 'P0020011']
vap_column_names = ['P0040001', 'P0040002', 'P0040005', 'P0040006', 'P0040007',
                    'P0040008', 'P0040009', 'P0040010', 'P0040011']

# Aggregates specific columns from population and VAP DataFrames into the election DataFrame
for name in pop_column_names:
    election_df[name] = population_df[name].groupby(blocks_to_precincts_assignment).sum()
for name in vap_column_names:
    election_df[name] = vap_df[name].groupby(vap_blocks_to_precincts_assignment).sum()

# Prints sums for verification of assignments
print(population_df['P0020001'].sum())
print(election_df['P0020001'].sum())
print(vap_df['P0040001'].sum())
print(election_df['P0040001'].sum())

# Begins precincts to districts assignment
print("start precincts to district assignment")
precincts_to_districts_assignment = maup.assign(election_df.geometry, district_df.geometry)
print("finish precincts to district assignment")

# Updates precincts to district mapping in the election DataFrame
district_col_name = "DISTRICTN"
election_df["SSD"] = precincts_to_districts_assignment

print(set(election_df["SSD"]))
for precinct_index in range(len(election_df)):
    election_df.at[precinct_index, "SSD"] = district_df.at[election_df.at[precinct_index, "SSD"], district_col_name]
print(set(district_df[district_col_name]))
print(set(election_df["SSD"]))

# Renames the columns according to the dictionary
rename_dict = {'P0020001': 'TOTPOP', 'P0020002': 'HISP', 'P0020005': 'NH_WHITE', 'P0020006': 'NH_BLACK', 'P0020007': 'NH_AMIN',
                    'P0020008': 'NH_ASIAN', 'P0020009': 'NH_NHPI', 'P0020010': 'NH_OTHER', 'P0020011': 'NH_2MORE',
                    'P0040001': 'VAP', 'P0040002': 'HVAP', 'P0040005': 'WVAP', 'P0040006': 'BVAP', 'P0040007': 'AMINVAP',
                                        'P0040008': 'ASIANVAP', 'P0040009': 'NHPIVAP', 'P0040010': 'OTHERVAP', 'P0040011': '2MOREVAP',
                                        'G20PREDBID': 'G20PRED', 'G20PRERTRU': 'G20PRER', 'G20USSDDUR': 'G20USSD', 
                                        'G20USSRCUR': 'G20USSR'}

print(list(election_df.columns))

election_df.rename(columns=rename_dict, inplace = True)

print(list(election_df.columns))
# Drops unused columns from the election DataFrame
election_df.drop(columns=[ 'G20PRELJOR','G20PREGHAW','G20PREACAR','G20PRESLAR',  'G20USSIWIL', 'G20USSLMAL','G20USSGBLA'], inplace=True)
print(list(election_df.columns))

# Plots the updated election DataFrame
election_df.plot()

# Prints population totals for each district
print(election_df.loc[election_df["SSD"] == 1, "TOTPOP"].sum())
print(election_df.loc[election_df["SSD"] == 2, "TOTPOP"].sum())
pop_vals = [election_df.loc[election_df["SSD"] == n, "TOTPOP"].sum() for n in range(1, 59)]
print(pop_vals)

#Creates shp file
election_df.to_file("./IL/IL.shp")

#Reads shp to make GeoJSON
shp_file = gpd.read_file('./IL/IL.shp')

#Creates GeoJSON
shp_file.to_file('./IL/IL.geojson', driver='GeoJSON')

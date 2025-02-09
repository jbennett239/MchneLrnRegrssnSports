#This project will be able to take two teams, compare their current stats and return which team will win.
#By utilizing machine learning regression methods this model will continue to stay updated and better the prediction process as it gets updated.
#Let's import the necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#Import the data
from google.colab import files
uploaded = files.upload()

#Lets start with the NBA teams
NCAA_Stats = pd.read_csv('NCAAStats2125.csv')

NCAA_Stats.info()

NCAA_Stats.describe()

NCAA_Stats.head(365)

#Using dataframe NCAA_Stats: compare two teams for which is better

# Select the two teams to compare.  Replace 'Team1' and 'Team2' with the actual team names.
team1 = 'Florida Gators'
team2 = 'Auburn Tigers'

# Find the statistics for each team.
team1_stats = NCAA_Stats[NCAA_Stats['Team'] == team1]
team2_stats = NCAA_Stats[NCAA_Stats['Team'] == team2]

# Check if teams exist in the DataFrame
if team1_stats.empty or team2_stats.empty:
    print(f"Error: One or both of the specified teams ({team1}, {team2}) not found in the DataFrame.")
else:
# Select relevant statistics for comparison.
    stats_to_compare = ['PTS', 'FGM', 'FG%', '3PM', '3P%', 'REB', 'AST', 'STL', 'BLK', 'TO']

    print(f"Comparison between {team1} and {team2}:")
    for stat in stats_to_compare:
        team1_value = team1_stats[stat].iloc[0]
        team2_value = team2_stats[stat].iloc[0]
        print(f"{stat}: {team1} - {team1_value:.1f}, {team2} - {team2_value:.1f}")

# Determine the "better" team based on a combined metric (you can customize this).
# Example:  Calculate a weighted score for each team based on different stats.
    weights = {'PTS': 0.2, 'FGM': 0.15, 'FG%': 0.15, '3PM': 0.1, '3P%': 0.1, 'REB': 0.1, 'AST': 0.1, 'STL': 0.05, 'BLK': 0.05, 'TO': -0.1}  #Negative weight for turnovers

    team1_score = sum(team1_stats[stat].iloc[0] * weights[stat] for stat in stats_to_compare)
    team2_score = sum(team2_stats[stat].iloc[0] * weights[stat] for stat in stats_to_compare)

    print(f"\nOverall weighted score:")
    print(f"{team1}: {team1_score:.2f}")
    print(f"{team2}: {team2_score:.2f}")

    if team1_score > team2_score:
        print(f"\nBased on this weighted score, the {team1} are considered a better team.")
    elif team2_score > team1_score:
        print(f"\nBased on this weighted score, the {team2} are considered a better team.")
    else:
        print("\nBased on this weighted score, both teams are equally matched")

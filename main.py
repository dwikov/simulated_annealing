import pandas as pd
import numpy as np

from random import random
from haversine import haversine
import matplotlib.pyplot as plt

# Read the CSV file and extract the 30 most populated cities
df = pd.read_csv('city.csv')
top_cities = df.nlargest(30, 'population')

# Calculate the distance between two cities
def distance(city1, city2):
    coord1 = (city1['geo_lat'], city1['geo_lon'])
    coord2 = (city2['geo_lat'], city2['geo_lon'])
    return haversine(coord1, coord2)

# Calculate the total distance for a given path
def total_distance(path):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(path[i], path[i + 1])
    dist += distance(path[-1], path[0])
    return dist

# SA Algorithm
def simulated_annealing(cities, initial_temp, cooling_rate, num_iterations):
    current_path = cities.copy()
    current_distance = total_distance(current_path)
    best_path = current_path.copy()
    best_distance = current_distance
    history = [best_distance]

    for _ in range(num_iterations):
        temp = initial_temp * (cooling_rate ** _)
        proposed_path = proposal_policy(current_path)
        proposed_distance = total_distance(proposed_path)

        if proposed_distance < current_distance or np.exp(-(proposed_distance - current_distance) / temp) > random():
            current_path = proposed_path
            current_distance = proposed_distance

            if current_distance < best_distance:
                best_path = current_path.copy()
                best_distance = current_distance
                history.append(best_distance)

    return best_path, best_distance, history

# Proposal policy
def proposal_policy(path):
    new_path = path.copy()
    i, j = np.random.choice(len(new_path), 2, replace=False)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# Run the algorithm
initial_temperature = 1000
cooling_rate = 0.995
iterations = 10000

cities = top_cities.to_dict('records')

cooling_rates = [0.999, 0.995, 0.98]
colors = ['r', 'g', 'b']
labels = ['Slow Cooling', 'Medium Cooling', 'Fast Cooling']

plt.figure(figsize=(12, 6))

for i, cooling_rate in enumerate(cooling_rates):
    _, _, history = simulated_annealing(cities, initial_temperature, cooling_rate, iterations)
    plt.plot(history, color=colors[i], label=labels[i])

plt.xlabel("Iterations")
plt.ylabel("Best Distance")
plt.title("Convergence for Different Cooling Rates")
plt.legend()
plt.show()

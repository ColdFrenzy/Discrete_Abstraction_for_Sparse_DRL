""" This utility script simulates the movement of N random points 
towards a target position within a grid, given a specific speed.
It calculates the time taken for each point to reach the target and
creates a histogram of the fastest 3 times across multiple iterations.
This allows for selecting the optimal number of UAVs needed based on 
environmental specifications
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_time(start_pos, target_pos, speed):
    """Calculate the time for a point to move from start_pos to target_pos at a given speed."""
    distance = math.sqrt((start_pos[0] - target_pos[0])**2 + (start_pos[1] - target_pos[1])**2)
    time = distance / speed
    return time

def generate_random_points(grid_size, N):
    """Generate N random points within the grid."""
    points = []
    for _ in range(N):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        points.append((x, y))
    return points

def simulate(grid_size, target_pos, N, speed):
    """Simulate the movement of N random points towards the target position and return the last of the fastest 3 times."""
    points = generate_random_points(grid_size, N)
    times = []

    # Calculate time for each point
    for point in points:
        time = calculate_time(point, target_pos, speed)
        times.append(time)

    # Get the top 3 fastest times
    top_3_times = sorted(times)
    return top_3_times[2]

def create_histograms(grid_size, target_pos, N_list, speed, num_iterations=100):
    """Create and save histograms of the fastest 3 times for each N in N_list across multiple iterations."""
    for N in N_list:
        all_top_3_times = []

        for _ in range(num_iterations):
            top_3_times = simulate(grid_size, target_pos, N, speed)
            all_top_3_times.extend([top_3_times])

        # Plot the histogram
        plt.hist(all_top_3_times, bins=30, edgecolor='black')
        plt.title(f"Histogram of Top 3 Fastest Times (Grid Size: {grid_size}m^2, N: {N} points, Speed: {speed} m/s)")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')

        # Save the plot
        filename = f"histogram_N_{N}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Histogram saved as {filename}")

# Parameters
grid_size = 100  # Grid size in meters (you can adjust this)
target_pos = (2, 2)  # Target position (x, y)
N_list = [3, 5, 10, 20]  # List of N values
speed = 17  # Speed of the points in meters per second

# Run the simulation and create the histograms
create_histograms(grid_size, target_pos, N_list, speed, num_iterations=5000)

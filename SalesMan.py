import os
from mpi4py import MPI
import numpy as np
import random
import time
import itertools

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Predefined Difficulty Levels
PREDEFINED_OPTIONS = {
    1: {"cities": 5, "population_size": 50, "iterations": 100, "adjustment_rate": 0.1, "runtime": 30},
    10: {"cities": 20, "population_size": 200, "iterations": 1000, "adjustment_rate": 0.15, "runtime": 120},
    20: {"cities": 50, "population_size": 500, "iterations": 5000, "adjustment_rate": 0.2, "runtime": 300},
    30: {"cities": 75, "population_size": 800, "iterations": 8000, "adjustment_rate": 0.25, "runtime": 450},
    35: {"cities": 100, "population_size": 1000, "iterations": 10000, "adjustment_rate": 0.3, "runtime": 600},
}

# Extend predefined options to fill intermediate levels
for level in range(1, 36):
    if level not in PREDEFINED_OPTIONS:
        lower = max(k for k in PREDEFINED_OPTIONS if k <= level)
        higher = min(k for k in PREDEFINED_OPTIONS if k > level)
        weight = (level - lower) / (higher - lower)
        PREDEFINED_OPTIONS[level] = {
            "cities": int(PREDEFINED_OPTIONS[lower]["cities"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["cities"] * weight),
            "population_size": int(PREDEFINED_OPTIONS[lower]["population_size"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["population_size"] * weight),
            "iterations": int(PREDEFINED_OPTIONS[lower]["iterations"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["iterations"] * weight),
            "adjustment_rate": round(PREDEFINED_OPTIONS[lower]["adjustment_rate"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["adjustment_rate"] * weight, 2),
            "runtime": int(PREDEFINED_OPTIONS[lower]["runtime"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["runtime"] * weight),
        }

# Clear Screen Function
def clear_screen():
    """Clear the terminal screen."""
    print("\033[H\033[J", end="")  # ANSI escape sequence for clearing screen

# User Configurations
def get_user_config():
    """Get runtime configurations interactively (only on rank 0)."""
    if rank == 0:
        clear_screen()
        print("=" * 70)
        print("🚛 Welcome to the Traveling Salesman Simulator 🚛")
        print("=" * 70)
        print("In this program, you will act as a traveling salesman attempting")
        print("to find the shortest possible route to visit a set of cities.")
        print("\nEach route optimization iteration represents an improvement")
        print("to the route, with the goal of minimizing the total distance traveled (in miles).")
        print("=" * 70)
        print("Choose how you'd like to configure your simulation:")
        print("1️⃣ Enter parameters manually.")
        print("2️⃣ Select a predefined difficulty level (1: simple, 35: extreme).")

        config_choice = input("Enter your choice (1 or 2): ").strip()
        while config_choice not in {"1", "2"}:
            config_choice = input("Invalid choice. Please enter 1 or 2: ").strip()

        if config_choice == "1":
            # Manual Configuration
            print("\n🛠️ Configuration:")
            print("Provide the following details for your traveling salesman simulation.\n")

            population_size = input("1️⃣ How large is the initial population? (default: 100): ").strip()
            population_size = int(population_size) if population_size else 100

            iterations = input("2️⃣ How many route optimization iterations should run? (default: 500): ").strip()
            iterations = int(iterations) if iterations else 500

            adjustment_rate = input("3️⃣ Route adjustment rate (chance of a random change) (default: 0.1): ").strip()
            adjustment_rate = float(adjustment_rate) if adjustment_rate else 0.1

            num_cities = input("4️⃣ Number of cities in the simulation? (default: 20): ").strip()
            num_cities = int(num_cities) if num_cities else 20
        else:
            # Predefined Difficulty
            print("\n🌟 Predefined Difficulty Levels 🌟")
            for level in range(1, 36):
                option = PREDEFINED_OPTIONS[level]
                print(f"{level:2}: {option['cities']} cities, {option['population_size']} population, "
                      f"{option['iterations']} route optimization iterations, adjustment rate {option['adjustment_rate']}")
            difficulty = int(input("\nEnter a difficulty level (1-35): ").strip())
            while difficulty not in range(1, 36):
                difficulty = int(input("Invalid level. Please enter a number between 1 and 35: ").strip())

            config = PREDEFINED_OPTIONS[difficulty]
            population_size = config["population_size"]
            iterations = config["iterations"]
            adjustment_rate = config["adjustment_rate"]
            num_cities = config["cities"]

        print("\n✨ Let the journey begin! ✨")
    else:
        population_size, iterations, adjustment_rate, num_cities = None, None, None, None

    # Broadcast settings to all ranks
    population_size = comm.bcast(population_size, root=0)
    iterations = comm.bcast(iterations, root=0)
    adjustment_rate = comm.bcast(adjustment_rate, root=0)
    num_cities = comm.bcast(num_cities, root=0)

    return population_size, iterations, adjustment_rate, num_cities

# Fitness Function (TSP Distance)
def fitness_function(route):
    """Calculate the total distance of the TSP route."""
    distance = 0.0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(CITIES[route[i]] - CITIES[route[i + 1]])
    distance += np.linalg.norm(CITIES[route[-1]] - CITIES[route[0]])  # Return to start
    return distance

# Generate Initial Population
def initialize_population(pop_size, num_cities):
    """Create initial population as random permutations of cities."""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Mutation Operator
def mutate(route):
    """Swap two cities in the route."""
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]

# Crossover Operator
def crossover(parent1, parent2):
    """Perform ordered crossover."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]

    # Fill remaining slots with parent2
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = gene
    return child

# Evolve Population
def evolve_population(population, adjustment_rate):
    """Evolve the population using selection, crossover, and mutation."""
    new_population = []

    # Tournament selection
    def select_parent():
        """Select a parent using tournament selection."""
        sample_size = min(5, len(population))  # Ensure sample size is valid
        return min(random.sample(population, sample_size), key=fitness_function)

    # Generate new population
    for _ in range(len(population)):
        parent1 = select_parent()
        parent2 = select_parent()
        child = crossover(parent1, parent2)
        if random.random() < adjustment_rate:
            mutate(child)
        new_population.append(child)
    return new_population

# Main Program
if __name__ == "__main__":
    # User-configurable settings
    population_size, iterations, adjustment_rate, num_cities = get_user_config()

    # Initialize city coordinates
    global CITIES
    CITIES = np.random.rand(num_cities, 2) * 100  # Randomly generate cities

    # Divide population among nodes
    local_population_size = population_size // size
    if rank < population_size % size:
        local_population_size += 1

    # Initialize local population
    population = initialize_population(local_population_size, num_cities)

    # Evolutionary Algorithm
    start_time = time.time()
    best_fitness = float("inf")
    best_route = None

    for iteration in range(iterations):
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)

        # Evolve local population
        population = evolve_population(population, adjustment_rate)

        # Find best in local population
        local_best = min(population, key=fitness_function)
        local_best_fitness = fitness_function(local_best)

        # Share best results among nodes
        all_best = comm.allgather((local_best_fitness, local_best))
        global_best_fitness, global_best = min(all_best, key=lambda x: x[0])

        # Update global best
        if global_best_fitness < best_fitness:
            best_fitness = global_best_fitness
            best_route = global_best

        # Display progress
        if rank == 0:
            print(f"\r🌟 Evolution Cycle {iteration + 1}/{iterations}: Shortest Route = {best_fitness:.2f} miles | Time Elapsed: {mins:02}:{secs:02}", end="")

    # Final Output
    if rank == 0:
        print("\n\n🎉 [RESULT] Shortest Route Found 🎉")
        print("This route represents the shortest distance to visit all cities and return to the start.")
        print("Order of cities visited:")
        print(" -> ".join(map(str, best_route)))
        print(f"\nTotal Distance: {best_fitness:.2f} miles")
        print("\n[INFO] Simulation complete. Thank you for using the Traveling Salesman Simulator!")
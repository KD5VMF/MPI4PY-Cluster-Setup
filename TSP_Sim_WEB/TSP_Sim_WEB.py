# tsp_simulation_dashboard.py

import os
import numpy as np
import random
import time
import threading
from mpi4py import MPI
from flask import Flask, jsonify, render_template_string, request
from werkzeug.serving import make_server
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Predefined Difficulty Levels (15 Levels)
PREDEFINED_OPTIONS = {
    1: {"cities": 5, "population_size": 50, "iterations": 100, "adjustment_rate": 0.1},
    4: {"cities": 10, "population_size": 100, "iterations": 500, "adjustment_rate": 0.12},
    7: {"cities": 15, "population_size": 150, "iterations": 1000, "adjustment_rate": 0.14},
    10: {"cities": 20, "population_size": 200, "iterations": 1500, "adjustment_rate": 0.16},
    13: {"cities": 25, "population_size": 250, "iterations": 2000, "adjustment_rate": 0.18},
    16: {"cities": 30, "population_size": 300, "iterations": 2500, "adjustment_rate": 0.2},
    19: {"cities": 35, "population_size": 350, "iterations": 3000, "adjustment_rate": 0.22},
    22: {"cities": 40, "population_size": 400, "iterations": 3500, "adjustment_rate": 0.24},
    25: {"cities": 45, "population_size": 450, "iterations": 4000, "adjustment_rate": 0.26},
    28: {"cities": 50, "population_size": 500, "iterations": 4500, "adjustment_rate": 0.28},
    31: {"cities": 55, "population_size": 550, "iterations": 5000, "adjustment_rate": 0.3},
    34: {"cities": 60, "population_size": 600, "iterations": 5500, "adjustment_rate": 0.32},
    37: {"cities": 165, "population_size": 1650, "iterations": 6000, "adjustment_rate": 0.34},
    40: {"cities": 270, "population_size": 1700, "iterations": 16500, "adjustment_rate": 0.36},
    43: {"cities": 375, "population_size": 1750, "iterations": 17000, "adjustment_rate": 0.38},
    46: {"cities": 400, "population_size": 1800, "iterations": 25000, "adjustment_rate": 0.1},
}

# Extend predefined options to fill intermediate levels through interpolation
for level in range(1, 47):
    if level not in PREDEFINED_OPTIONS:
        lower_levels = [k for k in PREDEFINED_OPTIONS.keys() if k < level]
        higher_levels = [k for k in PREDEFINED_OPTIONS.keys() if k > level]
        if lower_levels and higher_levels:
            lower = max(lower_levels)
            higher = min(higher_levels)
            weight = (level - lower) / (higher - lower)
            PREDEFINED_OPTIONS[level] = {
                "cities": int(PREDEFINED_OPTIONS[lower]["cities"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["cities"] * weight),
                "population_size": int(PREDEFINED_OPTIONS[lower]["population_size"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["population_size"] * weight),
                "iterations": int(PREDEFINED_OPTIONS[lower]["iterations"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["iterations"] * weight),
                "adjustment_rate": round(PREDEFINED_OPTIONS[lower]["adjustment_rate"] * (1 - weight) + PREDEFINED_OPTIONS[higher]["adjustment_rate"] * weight, 2),
            }

# --------------- Flask Setup (Only on Rank 0) ---------------
app = Flask(__name__) if rank == 0 else None

# Shared simulation data
simulation_data = {
    "running": False,
    "year": 0,
    "stats": {
        "steps": 0,
        "best_fitness": float("inf"),
        "elapsed_time": "00:00",
        "population_size": 100,
        "num_cities": 20,
        "adjustment_rate": 0.1,
        "iterations": 1000,
    },
    "best_route": [],
    "cities": [],
    "recent_events": [],
    "project_info": {
        "title": "Traveling Salesman Simulator",
        "description": "An MPI-based evolutionary algorithm simulator to find the shortest possible route visiting a set of cities.",
        "creator": "Chat-GPR 4o",
        "version": "1.0",
        "license": "MIT",
    }
}

# Lock for thread-safe operations
data_lock = threading.Lock()

# HTML Template for the dashboard with Bootstrap and Chart.js
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ project_info.title }} Dashboard</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .card { margin-bottom: 20px; }
        canvas { width: 100% !important; max-height: 500px !important; }
        .btn-group { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">{{ project_info.title }} Dashboard</h1>
        
        <!-- Project Information -->
        <div class="card">
            <div class="card-header">
                About This Project
            </div>
            <div class="card-body">
                <h5 class="card-title">{{ project_info.title }}</h5>
                <p class="card-text">{{ project_info.description }}</p>
                <p><strong>Creator:</strong> {{ project_info.creator }}</p>
                <p><strong>Version:</strong> {{ project_info.version }}</p>
                <p><strong>License:</strong> {{ project_info.license }}</p>
            </div>
        </div>
        
        <!-- Difficulty Selection -->
        <div class="card">
            <div class="card-header">
                Select Difficulty Level
            </div>
            <div class="card-body">
                <form id="difficulty-form">
                    <div class="form-group">
                        <label for="difficulty">Difficulty Level (1 - 46):</label>
                        <select class="form-control" id="difficulty" name="difficulty">
                            {% for level in predefined_levels %}
                                <option value="{{ level }}">{{ level }} - {{ predefined_options[level].cities }} cities, Population: {{ predefined_options[level].population_size }}, Iterations: {{ predefined_options[level].iterations }}, Adjustment Rate: {{ predefined_options[level].adjustment_rate }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary">Set Difficulty</button>
                </form>
            </div>
        </div>
        
        <!-- Control Buttons -->
        <div class="btn-group" role="group" aria-label="Control Buttons">
            <button type="button" class="btn btn-success" onclick="startSimulation()">Start Simulation</button>
            <button type="button" class="btn btn-danger" onclick="stopSimulation()">Stop Simulation</button>
            <button type="button" class="btn btn-warning" onclick="resetSimulation()">Reset Simulation</button>
        </div>
        
        <!-- Simulation Metrics -->
        <div class="row">
            <div class="col-md-3">
                <div class="card text-white bg-primary">
                    <div class="card-body">
                        <h5 class="card-title">Iterations Completed</h5>
                        <p class="card-text" id="stat-steps">0</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-success">
                    <div class="card-body">
                        <h5 class="card-title">Best Fitness</h5>
                        <p class="card-text" id="stat-best-fitness">∞</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-info">
                    <div class="card-body">
                        <h5 class="card-title">Elapsed Time</h5>
                        <p class="card-text" id="stat-elapsed-time">00:00</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-warning">
                    <div class="card-body">
                        <h5 class="card-title">Number of Cities</h5>
                        <p class="card-text" id="stat-num-cities">0</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Best Route Visualization -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        Best Route Visualization
                    </div>
                    <div class="card-body">
                        <canvas id="routeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Events -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        Recent Events
                    </div>
                    <div class="card-body">
                        <ul id="recent-events">
                            <!-- Events will be populated here -->
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- JavaScript Libraries -->
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    
    <script>
        // Initialize Route Chart as a Scatter Plot with Lines
        var ctx = document.getElementById('routeChart').getContext('2d');
        var routeChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Best Route',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 1)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    showLine: true,
                    fill: false,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                }]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Best Route Visualization'
                },
                tooltips: {
                    mode: 'index',
                    intersect: false,
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                },
                scales: {
                    xAxes: [{
                        type: 'linear',
                        position: 'bottom',
                        scaleLabel: {
                            display: true,
                            labelString: 'X Coordinate'
                        },
                        ticks: {
                            min: 0,
                            max: 100
                        }
                    }],
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'Y Coordinate'
                        },
                        ticks: {
                            min: 0,
                            max: 100
                        }
                    }]
                }
            }
        });
        
        // Function to update data
        function updateData() {
            $.ajax({
                url: '/api/data',
                method: 'GET',
                success: function(data) {
                    // Update Metrics
                    $('#stat-steps').text(data.stats.steps);
                    $('#stat-best-fitness').text(data.stats.best_fitness === Infinity ? '∞' : data.stats.best_fitness.toFixed(2));
                    $('#stat-elapsed-time').text(data.stats.elapsed_time);
                    $('#stat-num-cities').text(data.stats.num_cities);
                    
                    // Update Route Chart
                    var routeData = data.best_route.map(function(city) {
                        return {x: city[0], y: city[1]};
                    });
                    // Close the loop by returning to the start
                    if (routeData.length > 0) {
                        routeData.push(routeData[0]);
                    }
                    routeChart.data.datasets[0].data = routeData;
                    routeChart.update();
                    
                    // Update Recent Events
                    var eventsList = $('#recent-events');
                    eventsList.empty();
                    data.recent_events.slice(-10).reverse().forEach(function(event) {
                        eventsList.append('<li>' + event + '</li>');
                    });
                },
                error: function(error) {
                    console.error("Error fetching data:", error);
                }
            });
        }
        
        // Start Simulation
        function startSimulation() {
            $.post('/start', function(response) {
                if (response.status === 'started') {
                    alert("Simulation started.");
                } else if (response.status === 'already_running') {
                    alert("Simulation is already running.");
                } else {
                    alert("Failed to start simulation.");
                }
            }).fail(function() {
                alert("Error communicating with server.");
            });
        }
        
        // Stop Simulation
        function stopSimulation() {
            $.post('/stop', function(response) {
                if (response.status === 'stopped') {
                    alert("Simulation stopped.");
                } else if (response.status === 'not_running') {
                    alert("Simulation is not running.");
                } else {
                    alert("Failed to stop simulation.");
                }
            }).fail(function() {
                alert("Error communicating with server.");
            });
        }
        
        // Reset Simulation
        function resetSimulation() {
            $.post('/reset', function(response) {
                if (response.status === 'reset') {
                    alert("Simulation reset.");
                    updateData();
                } else {
                    alert("Failed to reset simulation.");
                }
            }).fail(function() {
                alert("Error communicating with server.");
            });
        }
        
        // Set Difficulty Level
        $('#difficulty-form').on('submit', function(e) {
            e.preventDefault();
            var selectedDifficulty = $('#difficulty').val();
            $.post('/set_difficulty', {difficulty: selectedDifficulty}, function(response) {
                if (response.status === 'difficulty_set') {
                    alert("Difficulty level set to " + selectedDifficulty + ".");
                    updateData();
                } else if (response.status === 'invalid_difficulty') {
                    alert("Invalid difficulty level selected.");
                } else {
                    alert("Failed to set difficulty level.");
                }
            }).fail(function() {
                alert("Error communicating with server.");
            });
        });
        
        // Initial Data Load
        updateData();
        
        // Periodically update data every 5 seconds
        setInterval(updateData, 5000);
    </script>
</body>
</html>
"""

# Server class to run Flask server in a separate thread
class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('0.0.0.0', 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()
    
    def run(self):
        try:
            self.server.serve_forever()
        except Exception as e:
            logging.error(f"Flask server encountered an error: {e}")
    
    def shutdown(self):
        self.server.shutdown()

# Simulation Control Flags
simulation_running = False
simulation_stop_event = threading.Event()
simulation_thread = None
server_thread = None

# Fitness Function and Genetic Operators
def fitness_function(route):
    """Calculate the total distance of the TSP route."""
    try:
        distance = 0.0
        for i in range(len(route) - 1):
            # Validate indices
            if route[i] >= simulation_data["stats"]["num_cities"] or route[i+1] >= simulation_data["stats"]["num_cities"]:
                raise IndexError(f"Route index out of bounds: {route[i]}, {route[i+1]} with num_cities={simulation_data['stats']['num_cities']}")
            distance += np.linalg.norm(CITIES[route[i]] - CITIES[route[i + 1]])
        # Validate last index with first
        if route[-1] >= simulation_data["stats"]["num_cities"] or route[0] >= simulation_data["stats"]["num_cities"]:
            raise IndexError(f"Route index out of bounds: {route[-1]}, {route[0]} with num_cities={simulation_data['stats']['num_cities']}")
        distance += np.linalg.norm(CITIES[route[-1]] - CITIES[route[0]])  # Return to start
        return distance
    except IndexError as e:
        logging.error(f"IndexError in fitness_function: {e}")
        return float("inf")  # Assign a very bad fitness if there's an error

def initialize_population(pop_size, num_cities):
    """Create initial population as random permutations of cities."""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def mutate(route):
    """Swap two cities in the route."""
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]

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

def validate_population(population, num_cities):
    """Validate that each route is a valid permutation of city indices."""
    valid = True
    for route in population:
        if len(route) != num_cities:
            logging.error(f"Invalid route length: {len(route)}. Expected: {num_cities}")
            valid = False
            break
        if sorted(route) != list(range(num_cities)):
            logging.error(f"Invalid route permutation: {route}")
            valid = False
            break
    if not valid:
        logging.error("Population validation failed.")
    return valid

def evolve_population(population, adjustment_rate):
    """Evolve the population using selection, crossover, and mutation."""
    new_population = []

    # Tournament selection
    def select_parent():
        """Select a parent using tournament selection."""
        sample_size = min(5, len(population))  # Ensure sample size is valid
        try:
            sampled = random.sample(population, sample_size)
        except ValueError as e:
            logging.error(f"Error in select_parent: {e}")
            sampled = population  # Fallback to entire population
        return min(sampled, key=fitness_function)

    try:
        # Generate new population
        for _ in range(len(population)):
            parent1 = select_parent()
            parent2 = select_parent()
            child = crossover(parent1, parent2)
            if random.random() < adjustment_rate:
                mutate(child)
            # Validate child
            with data_lock:
                num_cities = simulation_data["stats"]["num_cities"]
            if len(child) != num_cities:
                logging.error(f"Invalid child length: {len(child)}. Expected: {num_cities}")
                continue  # Skip invalid child
            if sorted(child) != list(range(num_cities)):
                logging.error(f"Invalid child permutation: {child}")
                continue  # Skip invalid child
            new_population.append(child)
    except Exception as e:
        logging.error(f"Error in evolve_population: {e}")
        # In case of error, retain the old population
        return population
    return new_population

def run_simulation(population_size, num_cities, adjustment_rate, iterations=1000):
    global simulation_data, simulation_running
    try:
        population = initialize_population(population_size, num_cities)
        best_fitness = float("inf")
        best_route = []
        start_time = time.time()

        for iteration in range(1, iterations + 1):
            if simulation_stop_event.is_set():
                logging.info("[Simulation] Stop signal received. Terminating simulation.")
                break

            # Evolve population
            population = evolve_population(population, adjustment_rate)

            # Validate population
            with data_lock:
                num_cities = simulation_data["stats"]["num_cities"]
            if not validate_population(population, num_cities):
                logging.error("[Simulation] Population is invalid. Resetting population.")
                population = initialize_population(simulation_data["stats"]["population_size"], num_cities)
                continue

            # Find best in current population
            current_best = min(population, key=fitness_function)
            current_fitness = fitness_function(current_best)

            # Update best fitness and route
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_route = current_best.copy()
                elapsed_time = time.time() - start_time
                mins, secs = divmod(int(elapsed_time), 60)
                with data_lock:
                    simulation_data["stats"]["best_fitness"] = best_fitness
                    simulation_data["best_route"] = [CITIES[city].tolist() for city in best_route]
                    simulation_data["stats"]["steps"] = iteration
                    simulation_data["stats"]["elapsed_time"] = f"{mins:02}:{secs:02}"
                    simulation_data["recent_events"].append(f"New best fitness: {best_fitness:.2f} at iteration {iteration}")

            # Update elapsed time
            elapsed_time = time.time() - start_time
            mins, secs = divmod(int(elapsed_time), 60)
            with data_lock:
                simulation_data["stats"]["elapsed_time"] = f"{mins:02}:{secs:02}"

            # Optional: Sleep to simulate runtime
            time.sleep(0.01)  # Reduced sleep time for faster simulation

    except Exception as e:
        logging.error(f"Exception in run_simulation: {e}")
    finally:
        with data_lock:
            simulation_running = False
            simulation_data["running"] = False
        logging.info("[Simulation] Simulation thread has terminated.")

# Flask Routes (Only on Rank 0)
if rank == 0:
    @app.route("/")
    def index():
        """
        Serve the dashboard with dynamic data.
        """
        with data_lock:
            rendered_html = render_template_string(
                html_template,
                stats=simulation_data["stats"],
                best_route=simulation_data["best_route"],
                recent_events=simulation_data["recent_events"],
                project_info=simulation_data["project_info"],
                predefined_levels=sorted(PREDEFINED_OPTIONS.keys()),
                predefined_options=PREDEFINED_OPTIONS
            )
        return rendered_html

    @app.route("/api/data")
    def api_data():
        """
        Provide simulation data in JSON format for AJAX requests.
        """
        with data_lock:
            response = {
                "year": simulation_data["year"],
                "stats": simulation_data["stats"],
                "best_route": simulation_data["best_route"],
                "cities": simulation_data["cities"],
                "recent_events": simulation_data["recent_events"],
            }
        return jsonify(response)

    @app.route("/start", methods=["POST"])
    def start():
        """
        Start the simulation.
        """
        global simulation_thread, simulation_running
        need_to_start = False
        with data_lock:
            if not simulation_data["running"]:
                simulation_data["running"] = True
                simulation_stop_event.clear()
                need_to_start = True
                simulation_thread = threading.Thread(
                    target=run_simulation,
                    args=(
                        simulation_data["stats"]["population_size"],
                        simulation_data["stats"]["num_cities"],
                        simulation_data["stats"]["adjustment_rate"],
                        simulation_data["stats"]["iterations"]
                    )
                )
        if need_to_start:
            simulation_thread.start()
            with data_lock:
                simulation_data["recent_events"].append("Simulation started.")
            logging.info("[Rank 0] Simulation started.")
            return jsonify({"status": "started"})
        else:
            logging.warning("[Rank 0] Attempted to start simulation, but it's already running.")
            return jsonify({"status": "already_running"})

    @app.route("/stop", methods=["POST"])
    def stop():
        """
        Stop the simulation.
        """
        global simulation_thread, simulation_running
        need_to_stop = False
        with data_lock:
            if simulation_data["running"]:
                simulation_stop_event.set()
                need_to_stop = True
        if need_to_stop and simulation_thread is not None:
            simulation_thread.join(timeout=10)
            with data_lock:
                if simulation_thread.is_alive():
                    logging.error("[Rank 0] Simulation thread did not terminate within timeout.")
                else:
                    simulation_data["running"] = False
                    simulation_data["recent_events"].append("Simulation stopped.")
                    logging.info("[Rank 0] Simulation stopped.")
                    simulation_thread = None
        else:
            logging.warning("[Rank 0] Attempted to stop simulation, but it's not running.")
        return jsonify({"status": "stopped"})

    @app.route("/reset", methods=["POST"])
    def reset():
        """
        Reset the simulation.
        """
        global simulation_thread, simulation_running, CITIES
        need_to_stop = False
        with data_lock:
            if simulation_data["running"]:
                simulation_stop_event.set()
                need_to_stop = True
        if need_to_stop and simulation_thread is not None:
            simulation_thread.join(timeout=10)
            with data_lock:
                if simulation_thread.is_alive():
                    logging.error("[Rank 0] Simulation thread did not terminate within timeout.")
                else:
                    simulation_data["running"] = False
                    simulation_data["recent_events"].append("Simulation stopped for reset.")
                    logging.info("[Rank 0] Simulation stopped for reset.")
                    simulation_thread = None
        with data_lock:
            # Reset simulation data
            simulation_data["year"] = 0
            simulation_data["stats"]["steps"] = 0
            simulation_data["stats"]["best_fitness"] = float("inf")
            simulation_data["best_route"] = []
            simulation_data["stats"]["elapsed_time"] = "00:00"
            simulation_data["recent_events"] = ["Simulation reset."]
            # Reinitialize city coordinates
            CITIES = np.random.rand(simulation_data["stats"]["num_cities"], 2) * 100
            simulation_data["cities"] = CITIES.tolist()
            simulation_data["recent_events"].append("City coordinates reinitialized.")
            logging.info("[Rank 0] Simulation data reset.")
        return jsonify({"status": "reset"})

    @app.route("/set_difficulty", methods=["POST"])
    def set_difficulty():
        """
        Set the difficulty level.
        """
        global simulation_thread, simulation_running, PREDEFINED_OPTIONS, CITIES
        try:
            difficulty = int(request.form.get('difficulty', 1))
            if difficulty not in PREDEFINED_OPTIONS:
                logging.error(f"[Rank 0] Invalid difficulty level received: {difficulty}")
                return jsonify({"status": "invalid_difficulty"})
            
            need_to_stop = False
            with data_lock:
                # If simulation is running, stop it before changing difficulty
                if simulation_data["running"]:
                    simulation_stop_event.set()
                    need_to_stop = True
            if need_to_stop and simulation_thread is not None:
                simulation_thread.join(timeout=10)
                with data_lock:
                    if simulation_thread.is_alive():
                        logging.error("[Rank 0] Simulation thread did not terminate within timeout.")
                    else:
                        simulation_data["running"] = False
                        simulation_data["recent_events"].append("Simulation stopped for difficulty change.")
                        logging.info("[Rank 0] Simulation stopped for difficulty change.")
                        simulation_thread = None
            
            with data_lock:
                # Update simulation parameters
                simulation_data["stats"]["num_cities"] = PREDEFINED_OPTIONS[difficulty]["cities"]
                simulation_data["stats"]["population_size"] = PREDEFINED_OPTIONS[difficulty]["population_size"]
                simulation_data["stats"]["iterations"] = PREDEFINED_OPTIONS[difficulty]["iterations"]
                simulation_data["stats"]["adjustment_rate"] = PREDEFINED_OPTIONS[difficulty]["adjustment_rate"]
                
                # Reinitialize city coordinates based on new number of cities
                CITIES = np.random.rand(simulation_data["stats"]["num_cities"], 2) * 100
                simulation_data["cities"] = CITIES.tolist()
                simulation_data["best_route"] = []
                simulation_data["stats"]["best_fitness"] = float("inf")
                simulation_data["stats"]["steps"] = 0
                simulation_data["stats"]["elapsed_time"] = "00:00"
                simulation_data["recent_events"].append(f"Difficulty level set to {difficulty}. Simulation parameters updated.")
                logging.info(f"[Rank 0] Difficulty level set to {difficulty}. Simulation parameters updated.")
            
            return jsonify({"status": "difficulty_set"})
        
        except Exception as e:
            logging.error(f"[Rank 0] Exception in set_difficulty: {e}")
            return jsonify({"status": "error", "message": str(e)})

    # Initialize city coordinates
    CITIES = np.random.rand(simulation_data["stats"]["num_cities"], 2) * 100  # Randomly generate cities
    simulation_data["cities"] = CITIES.tolist()

    # Start Flask server in a separate thread
    try:
        server = make_server('0.0.0.0', 5000, app)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        logging.info("[Rank 0] Flask server started on port 5000.")
    except Exception as e:
        logging.error(f"[Rank 0] Failed to start Flask server: {e}")

# Worker Ranks
else:
    def worker_loop():
        """
        Worker loop to handle simulation tasks.
        """
        while True:
            try:
                # Receive task from master
                task = comm.recv(source=0, tag=100)
                if task is None:
                    # Termination signal
                    logging.info(f"[Rank {rank}] Received termination signal. Exiting.")
                    break

                # Perform simulation task (e.g., evolve population)
                population = task
                adjustment_rate = 0.1  # Default adjustment rate; can be updated as needed
                population = evolve_population(population, adjustment_rate)

                # Find best in local population
                local_best = min(population, key=fitness_function)
                local_best_fitness = fitness_function(local_best)

                # Send best fitness and route back to master
                comm.send({
                    "best_fitness": local_best_fitness,
                    "best_route": local_best,
                    "recent_events": [f"Rank {rank}: New local best fitness: {local_best_fitness:.2f}"]
                }, dest=0, tag=101)
            
            except Exception as e:
                logging.error(f"[Rank {rank}] Exception in worker_loop: {e}")
                break

    # Start worker loop
    worker_thread = threading.Thread(target=worker_loop)
    worker_thread.start()

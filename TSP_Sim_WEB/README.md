
# Traveling Salesman Problem (TSP) Simulation Dashboard

This project is an MPI-based evolutionary algorithm simulation to solve the Traveling Salesman Problem (TSP). It includes a Flask-powered dashboard for live visualization and control, supporting scalable computation using `mpi4py`.

## Features

- **Interactive Dashboard**: Control simulation parameters and visualize results in real time.
- **MPI-Based Scalability**: Distribute the computational workload across multiple nodes using MPI.
- **Dynamic TSP Parameters**: Adjust the number of cities, population size, and more to explore different complexities.
- **Genetic Algorithm**: Uses a genetic algorithm for evolving solutions to the TSP.
- **Detailed Metrics**: Track metrics like elapsed time, best fitness, iterations, and city coordinates.
- **Dynamic Visualization**: Real-time updates to the best route visualization using Chart.js.
- **Difficulty Levels**: Predefined difficulty levels (1-46) with parameters interpolated for intermediate levels.

## Requirements

- Python 3.7+
- Required Python packages:
  - `numpy`
  - `flask`
  - `mpi4py`
  - `werkzeug`
  - `chart.js`
  - `jquery`
  - `bootstrap`

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure MPI is installed and configured on your system.

## Usage

### Running the Simulation

1. **Start the Program**:
   On the master node (rank 0), start the program using:
   ```bash
   mpirun -np <number_of_processes> python tsp_simulation_dashboard.py
   ```

2. **Access the Dashboard**:
   Open a web browser and navigate to:
   ```
   http://<master_node_ip>:5000
   ```

3. **Control Simulation**:
   Use the dashboard to:
   - Start/stop/reset the simulation.
   - Adjust difficulty levels dynamically.

4. **Monitor Metrics**:
   View live metrics, such as:
   - Iterations completed
   - Best fitness
   - Elapsed time
   - Number of cities

5. **Visualize Results**:
   - Track the best route dynamically on the visualization chart.

### Notes

- The simulation runs indefinitely unless manually stopped.
- The elapsed time card changes its background color to green when a new best fitness is found and reverts after a few seconds.

## Code Structure

- `tsp_simulation_dashboard.py`: Main program file containing all functionalities.
- Genetic Algorithm Components:
  - `initialize_population`: Randomly initializes the population.
  - `fitness_function`: Calculates the total distance of a route.
  - `mutate`: Mutates a route by swapping two cities.
  - `crossover`: Performs ordered crossover between two parents.
  - `evolve_population`: Evolves the population using genetic operations.

## Known Issues

- Ensure all nodes are reachable and properly configured for MPI.
- Invalid city coordinates or route indices can result in runtime errors.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License.

---

**Creator**: Chat-GPT 4o  
**Version**: 1.0  

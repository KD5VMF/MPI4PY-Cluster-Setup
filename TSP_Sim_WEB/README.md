
# 🌟 Traveling Salesman Simulator Dashboard 🌟

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![MPI](https://img.shields.io/badge/MPI-OpenMPI-blue)

## 🗺️ About the Project

**Traveling Salesman Simulator Dashboard** is an interactive, web-based simulator for solving the Traveling Salesman Problem (TSP) using an evolutionary algorithm. It leverages **MPI** for parallel computation and **Flask** for a responsive dashboard interface.

The simulation calculates the shortest possible route visiting a set of cities and returning to the start. The project is ideal for:

- Learning about evolutionary algorithms and optimization.
- Experimenting with MPI-based parallel programming.
- Visualizing TSP solutions interactively.

---

## ✨ Features

- 🖥️ **Web Dashboard**:
  - Monitor simulation metrics like steps, best fitness, and elapsed time.
  - Visualize the shortest route in real-time.
  - View recent events and simulation progress.
- ⚙️ **Difficulty Levels**:
  - Choose from **46 predefined difficulty levels**.
  - Difficulty levels range from simple (5 cities) to extreme (95 cities).
- 🎛️ **Controls**:
  - Start, stop, or reset the simulation directly from the web interface.
  - Update simulation parameters dynamically by selecting a difficulty level.
- 🔍 **Best Route Visualization**:
  - Real-time scatter plot with connecting lines for the best route.
- 🌐 **MPI Parallel Processing**:
  - Efficiently distributes computations across multiple processes.

---

## 🚀 Getting Started

Follow these instructions to set up and run the Traveling Salesman Simulator Dashboard on your machine.

### Prerequisites

- Python 3.12 or later
- MPI installed (e.g., OpenMPI)
- Python packages:
  - `mpi4py`
  - `numpy`
  - `flask`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tsp-simulator-dashboard.git
   cd tsp-simulator-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure MPI is installed:
   ```bash
   sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   ```

---

## ⚙️ Usage

### Running the Simulation

Use the following command to run the program with MPI:

```bash
mpirun -n <number_of_processes> python tsp_simulation_dashboard.py
```

- Replace `<number_of_processes>` with the desired number of processes.

### Accessing the Dashboard

1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Use the web interface to:
   - Start, stop, or reset the simulation.
   - Set the difficulty level.

---

## 📊 Simulation Metrics

The dashboard displays real-time metrics, including:
- **Iterations Completed**: Total optimization steps.
- **Best Fitness**: Shortest route distance found.
- **Elapsed Time**: Time since the simulation started.
- **Number of Cities**: Total cities in the current simulation.

---

## 🛠️ Built With

- **Python**: The core programming language.
- **Flask**: Web framework for the dashboard.
- **MPI**: Parallel computing framework for distributed processing.
- **Chart.js**: Interactive chart library for visualizations.

---

## 🤝 Contributing

Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📧 Contact

- **Creator**: [Chat-GPT 4o](mailto:your-email@example.com)
- **Project Link**: [GitHub Repository](https://github.com/your-username/tsp-simulator-dashboard)

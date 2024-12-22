
# Enhanced World Simulation

![License](https://img.shields.io/github/license/yourusername/EnhancedWorldSimulation)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![MPI](https://img.shields.io/badge/MPI-Implemented-brightgreen.svg)

![Simulation Overview](assets/simulation_overview.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Welcome to the **Enhanced World Simulation**, a sophisticated and dynamic simulation environment that models a complex world populated by multiple entities. These entities interact, gather resources, discover knowledge, create items, establish homes and towns, initiate space flights, and manage extensive infrastructure and societal structures. Leveraging **Reinforcement Learning (Deep Q-Networks)** and **MPI (Message Passing Interface)** for parallel processing, this simulation provides an intricate and scalable platform for studying emergent behaviors and societal evolution.

![Entities Interaction](assets/entities_interaction.png)

## Features

- **Diverse Entity Roles:** Entities can assume various roles such as Strategists, Engineers, Scientists, Leaders, Diplomats, and more, each with unique capabilities and behaviors.
  
- **Resource Management:** Entities gather and manage resources like food, gold, influence, and crafted goods to sustain and advance their operations.
  
- **Knowledge Discovery:** Through actions like exploring and experimenting, entities can uncover new knowledge that enhances their abilities and the world's infrastructure.
  
- **Infrastructure Development:** Entities can build and upgrade a wide range of infrastructures, including transportation systems, utilities, social services, and economic facilities.
  
- **Societal Structures:** Establish homes, found towns, form alliances, and manage social classes to create a thriving society.
  
- **Space Exploration:** Initiate space flights to expand the simulation beyond the planet, adding a futuristic dimension to the world.
  
- **Reinforcement Learning Integration:** Utilize DQN to enable entities to learn optimal actions based on their state and experiences.
  
- **Parallel Processing with MPI:** Scale the simulation efficiently across multiple nodes and processes, handling thousands of entities seamlessly.

![Infrastructure Development](assets/infrastructure_development.png)

## Architecture

The simulation is built upon a modular architecture that combines object-oriented programming with reinforcement learning techniques. Here's an overview of the core components:

![Architecture Diagram](assets/architecture_diagram.png)

- **Entity Class:** Represents individual entities with attributes like health, intelligence, resources, skills, and knowledge.
  
- **ActionModel (DQN):** A neural network model that predicts Q-values for possible actions, guiding entities' decision-making processes.
  
- **Prioritized Replay Buffer:** Stores experiences for training the DQN, prioritizing important transitions to enhance learning efficiency.
  
- **MPI Integration:** Distributes the simulation workload across multiple nodes and processes, enabling high scalability and performance.
  
- **Event Logger:** Records significant events and changes within the simulation for analysis and debugging.
  
- **Persistence Layer:** Handles saving and loading of the world state, global knowledge, model weights, and simulation parameters to ensure continuity between sessions.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **MPI Implementation:** Ensure you have an MPI implementation like [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/) installed.
- **Git:** For version control and repository management.

### Clone the Repository

```bash
git clone https://github.com/yourusername/EnhancedWorldSimulation.git
cd EnhancedWorldSimulation
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not provided, you can install the necessary packages manually:*

```bash
pip install mpi4py torch numpy
```

## Usage

### Running the Simulation

The simulation leverages MPI for parallel processing. To run the simulation across multiple nodes or processes, use the `mpirun` or `mpiexec` command.

```bash
mpirun --hostfile mpi_hosts -np 24 python3 GameSim50.py
```

- `--hostfile mpi_hosts`: Specifies the file containing the list of hostnames or IP addresses of the nodes.
- `-np 24`: Number of processes to run (adjust based on your setup).

**Note:** Ensure that passwordless SSH is set up between the nodes listed in `mpi_hosts` to allow seamless communication.

### Configuration

Upon running the simulation, you'll be prompted with the following options:

1. **Reset Simulation:** Choose to reset the simulation, which deletes existing saved data and initializes a new world.
   
2. **Simulation Years:** Enter the number of simulation years to run.

**Example Interaction:**

```plaintext
==========================================
      Welcome to the Enhanced World Simulation
==========================================

This simulation models a dynamic world with multiple entities that interact, gather resources, discover knowledge, create items,
establish homes and towns, initiate space flight, and manage extensive infrastructure and societal structures.
Reinforcement Learning (DQN) is utilized to optimize entity actions based on their states and experiences.

Saved Files:
1. 'world_save.json' - Stores the current state of all entities in the world.
2. 'global_knowledge.json' - Contains the collective knowledge discovered by entities.
3. 'model_weights.pt' - Saves the trained DQN model weights for future simulations.
4. 'state.json' - Stores the current simulation state including simulation year, training steps, epsilon, and beta.

Do you want to reset the simulation? This will delete existing saved data. (yes/no) [no]: no
Enter the number of years to run the simulation: 1000
[INFO] Simulation will run for 1000 years.
```

## Contributing

Contributions are welcome! If you'd like to enhance the simulation, fix bugs, or add new features, please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

Provide a clear description of your changes and the motivation behind them.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **[OpenAI](https://www.openai.com/):** For providing the foundational technologies and inspiration.
- **[PyTorch](https://pytorch.org/):** For the deep learning framework.
- **[mpi4py](https://mpi4py.readthedocs.io/en/stable/):** For MPI integration in Python.
- **Community Contributors:** Thanks to all the contributors who have helped in enhancing this simulation.

---

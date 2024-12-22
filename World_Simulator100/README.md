# World Simulator 100

Welcome to **World Simulator 100**! This project is an advanced simulation platform designed for building, observing, and analyzing dynamic virtual worlds. Using a combination of reinforcement learning (RL), multi-agent systems, and parallel computation with MPI, this program models complex interactions, resource management, and societal evolution.

---

![World Simulator 100](https://via.placeholder.com/800x400?text=World+Simulator+100+Banner)

## Features

- **Multi-Agent RL:** Each agent in the world uses reinforcement learning to make decisions based on its environment.
- **Dynamic Knowledge Sharing:** Entities discover and share knowledge dynamically.
- **Complex World Infrastructure:** Includes detailed roles, resources, and infrastructures like transportation, utilities, and more.
- **Parallel Simulation:** MPI enables scalability for large simulations across multiple processors or nodes.
- **Customizable Entities:** Define specific behaviors, roles, and attributes for each agent in the simulation.
- **Event-Driven Evolution:** Entities react to events, disasters, and opportunities in real-time.

---

## Table of Contents

1. [Introduction](#introduction)
2. [How It Works](#how-it-works)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Core Components](#core-components)
7. [Simulation Parameters](#simulation-parameters)
8. [Advanced Customization](#advanced-customization)
9. [Examples](#examples)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction

World Simulator 100 allows users to explore the behavior of entities in a simulated environment, complete with learning algorithms and complex interactions. From building cities to initiating space travel, this simulator provides endless possibilities for experimentation.

### Objectives

- Study emergent behaviors in simulated societies.
- Test reinforcement learning algorithms in dynamic environments.
- Analyze the impact of knowledge sharing and collaboration.
- Design infrastructure systems and measure their efficiency.

---

## How It Works

The simulation revolves around entities that:

1. **Perceive:** Gather data about their environment.
2. **Learn:** Use RL models to optimize decision-making.
3. **Act:** Execute actions like exploring, building, or trading.
4. **Collaborate:** Share resources and knowledge.

The environment evolves based on entity actions and external events, such as natural disasters or technological advancements.

---

## Getting Started

This guide assumes familiarity with Python and reinforcement learning concepts.

### Prerequisites

- Python 3.8 or higher.
- MPI library (`mpich` or equivalent).
- Essential Python libraries: `torch`, `mpi4py`, `numpy`, and `json`.

---

## Installation

### 1. Clone the Repository

```bash
$ git clone https://github.com/your-username/world-simulator-100.git
$ cd world-simulator-100
```

### 2. Install Dependencies

```bash
$ pip install -r requirements.txt
```

### 3. Setup MPI

For Ubuntu:

```bash
$ sudo apt-get install mpich
```

---

## Usage

### Running the Simulation

Run the program using MPI:

```bash
$ mpirun -n 4 python World_Simulator100.py
```

This initializes the simulation with 4 parallel processes. Modify `-n` based on your system.

### Save and Load States

- Save the simulation state:
  ```bash
  $ python World_Simulator100.py --save
  ```
- Load a previous simulation:
  ```bash
  $ python World_Simulator100.py --load
  ```

---

## Core Components

### Entities

Entities represent the core agents in the world. Each has attributes like:

- **Health**
- **Energy**
- **Resources**
- **Knowledge**
- **Skills**
- **Roles** (e.g., Farmer, Warrior, Scientist)

### Actions

Entities can perform actions such as:

- **Gathering:** Collect food, gold, or influence.
- **Building:** Construct homes, towns, or infrastructure.
- **Exploring:** Discover new lands or resources.
- **Trading:** Exchange goods with other entities.
- **Learning:** Enhance skills or acquire knowledge.

### Knowledge Sharing

Discoveries made by one entity can be shared globally, allowing for technological and cultural evolution.

---

## Simulation Parameters

| Parameter          | Description                                | Default Value |
|--------------------|--------------------------------------------|---------------|
| `NUM_ENTITIES`     | Number of entities to initialize           | `100`         |
| `GAMMA`            | Discount factor for RL                     | `0.99`        |
| `EPSILON`          | Initial exploration rate                   | `1.0`         |
| `LR`               | Learning rate for neural network training  | `0.001`       |
| `MAX_ITERATIONS`   | Maximum steps per simulation               | `10000`       |

---

## Advanced Customization

You can create entirely new simulations by customizing entities, roles, and actions.

### Adding a New Role

Define a new role with specific capabilities:

```python
class Merchant(Entity):
    def trade(self):
        # Custom trading logic
        pass
```

### Modifying Actions

Extend existing actions or create new ones:

```python
def explore_new_terrain(self):
    # Exploration logic
    pass
```

---

## Examples

### General Ways to Run World_Simulator100.py

Here are various ways to run the program, demonstrating its flexibility and utility:

- **Default Simulation:**
  ```bash
  $ python World_Simulator100.py
  ```
  This runs the simulation with default parameters.

- **Custom Number of Entities:**
  ```bash
  $ python World_Simulator100.py --entities 200
  ```
  Specify the number of entities in the world.

- **Set Maximum Iterations:**
  ```bash
  $ python World_Simulator100.py --iterations 5000
  ```
  Control the number of iterations for the simulation.

- **Enable Debug Mode:**
  ```bash
  $ python World_Simulator100.py --debug
  ```
  Outputs detailed logs for debugging.

- **Use MPI for Parallel Execution:**
  ```bash
  $ mpirun -n 4 python World_Simulator100.py
  ```
  Distributes the workload across 4 processes.

- **Enable Rapid Knowledge Sharing:**
  ```bash
  $ python World_Simulator100.py --share-knowledge
  ```
  Accelerates the dissemination of discovered knowledge.

- **Run with Custom Learning Rate and Exploration:**
  ```bash
  $ python World_Simulator100.py --lr 0.0005 --epsilon 0.2
  ```

- **Save Simulation State:**
  ```bash
  $ python World_Simulator100.py --save
  ```
  Saves the current world state to a file.

- **Load a Saved State:**
  ```bash
  $ python World_Simulator100.py --load
  ```
  Loads a previously saved simulation state.

- **Run with Energy-Based Decisions:**
  ```bash
  $ python World_Simulator100.py --energy-factor 1.2
  ```
  Alters how energy affects decision-making.

---

### Example 1: Basic Simulation

```bash
$ mpirun -n 4 python World_Simulator100.py --entities 50 --iterations 1000
```

### Example 2: Advanced RL Training

Train entities with a range of custom reinforcement learning parameters to experiment with different learning strategies and outcomes.

**Usage Examples:**

- Standard Learning Rate and Exploration:
  ```bash
  $ mpirun -n 8 python World_Simulator100.py --lr 0.0005 --epsilon 0.1
  ```

- High Learning Rate for Rapid Adaptation:
  ```bash
  $ mpirun -n 8 python World_Simulator100.py --lr 0.005 --epsilon 0.1
  ```

- Low Exploration for Stable Learning:
  ```bash
  $ mpirun -n 8 python World_Simulator100.py --lr 0.0005 --epsilon 0.01
  ```

- Adaptive Exploration with Decay:
  ```bash
  $ mpirun -n 8 python World_Simulator100.py --lr 0.0005 --epsilon_decay 0.995
  ```

Modify these parameters as needed to test the effects of learning rate (`--lr`) and exploration (`--epsilon` or `--epsilon_decay`) on the agents' behaviors.bash
$ mpirun -n 8 python World_Simulator100.py --lr 0.0005 --epsilon 0.1
```

### Example 3: Custom Knowledge Sharing

Enable rapid knowledge dissemination:

```bash
$ python World_Simulator100.py --share-knowledge
```

---

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For support, contact us at [support@worldsim100.com](mailto:support@worldsim100.com).

![Thank You](https://via.placeholder.com/600x150?text=Thank+You+for+Using+World+Simulator+100)

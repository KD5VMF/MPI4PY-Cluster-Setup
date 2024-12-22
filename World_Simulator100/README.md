# World Simulator 100

Welcome to **World Simulator 100**! This project is a detailed simulation platform designed for building complex worlds with entities, resources, and dynamic interactions driven by reinforcement learning and parallel computation using MPI.

---

![World Simulator 100](https://via.placeholder.com/800x400?text=World+Simulator+100+Banner)

## Features

- **Reinforcement Learning:** Built-in neural networks for decision-making.
- **Multi-Agent Simulation:** Entities interact and evolve within a virtual world.
- **Global Knowledge Sharing:** Distributed discovery and sharing of knowledge.
- **Extensive Roles and Infrastructure:** Simulate societies with various roles, infrastructures, and economic activities.
- **Scalable:** Supports parallel computation with MPI to handle large-scale simulations.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)

---

## Getting Started

World Simulator 100 is designed for researchers, hobbyists, and simulation enthusiasts. You can create complex societies, simulate their evolution, and analyze outcomes.

### Prerequisites

- Python 3.8 or higher
- `mpi4py`
- `torch`
- `numpy`
- Basic knowledge of Python and reinforcement learning is recommended.

---

## Installation

### 1. Clone the Repository

```bash
$ git clone https://github.com/your-username/world-simulator-100.git
$ cd world-simulator-100
```

### 2. Install Dependencies

Use the following command to install required dependencies:

```bash
$ pip install -r requirements.txt
```

Dependencies:
- mpi4py
- torch
- numpy

### 3. Setup MPI

Ensure MPI is installed on your system. For Ubuntu:

```bash
$ sudo apt-get install mpich
```

Verify installation:

```bash
$ mpirun --version
```

---

## Usage

### Running the Simulation

Launch the simulation using MPI to utilize parallel processing:

```bash
$ mpirun -n 4 python World_Simulator100.py
```

Here:
- `-n 4` specifies the number of processes.
- Replace `4` with the number of cores or nodes available.

### Simulation Overview

- **Entities:** Each entity has attributes such as health, mood, intelligence, and resources.
- **Actions:** Entities perform actions like gathering, building, exploring, and trading.
- **Knowledge Sharing:** Knowledge discovered by one entity can be shared globally.

### Example Output

```plaintext
[YEAR 2025] Entity_1 discovered advanced agriculture.
[YEAR 2026] Entity_2 built a solar farm.
[YEAR 2027] Natural disaster near (25, -14).
```

### Save and Load

- Save the simulation state:
  ```bash
  $ python World_Simulator100.py --save
  ```
- Load a previous simulation:
  ```bash
  $ python World_Simulator100.py --load
  ```

---

## Configuration

### Parameters

You can modify simulation parameters in the `World_Simulator100.py` file:

| Parameter      | Description                                | Default Value |
|----------------|--------------------------------------------|---------------|
| `NUM_ENTITIES` | Number of entities to initialize           | `100`         |
| `GAMMA`        | Discount factor for RL                     | `0.99`        |
| `EPSILON`      | Initial exploration rate                   | `1.0`         |
| `LR`           | Learning rate for neural network training  | `0.001`       |

### Customizing Entities

Entities can be customized with unique attributes:
- Roles: Farmer, Warrior, Scientist, etc.
- Specializations: Mining, Architecture, Diplomacy, etc.

Example snippet:

```python
entity = Entity(
    name="Entity_X",
    role="Scientist",
    health=100,
    location=(0, 0),
    specialization="Engineering"
)
```

---

## Contributing

We welcome contributions to improve and expand World Simulator 100! To contribute:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions or support, contact us at: [support@worldsim100.com](mailto:support@worldsim100.com)

---

![Thank You](https://via.placeholder.com/600x150?text=Thank+You+for+Using+World+Simulator+100)

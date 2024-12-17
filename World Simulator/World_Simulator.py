from mpi4py import MPI
import random
import time
import os
import json
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ========== MPI Setup ==========
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ================= Reinforcement Learning (RL) and Model Definitions =================

class ActionModel(nn.Module):
    """
    Neural Network model to predict Q-values for each possible action.
    """
    def __init__(self, input_dim=15, action_dim=20, version=2):
        """
        Initializes the ActionModel with specified input and action dimensions.

        Args:
            input_dim (int): Dimension of the input state vector.
            action_dim (int): Number of possible actions.
            version (int): Version number for the model architecture.
        """
        super(ActionModel, self).__init__()
        self.version = version
        self.fc1 = nn.Linear(input_dim, 128)  # Updated to match new state vector size
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

class ReplayBuffer:
    """
    Experience Replay Buffer to store and sample experiences for training the RL model.
    """
    def __init__(self, capacity=1000000):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        """
        Adds an experience to the buffer.

        Args:
            experience (tuple): A tuple of (state, action, reward, next_state, done).
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Returns the current size of the buffer.

        Returns:
            int: Number of experiences stored.
        """
        return len(self.buffer)

# Hyperparameters for DQN
GAMMA = 0.99          # Discount factor for future rewards
LR = 0.001            # Learning rate for optimizer
BATCH_SIZE = 64       # Batch size for training
TARGET_UPDATE = 1000  # Steps between target network updates
EPSILON_START = 0.1   # Initial exploration rate
EPSILON_DECAY = 0.99999 # Decay rate for exploration
EPSILON_MIN = 0.01    # Minimum exploration rate

class EventLogger:
    """
    Handles logging of significant events in the simulation.
    """
    def __init__(self, log_file="event_log.txt"):
        """
        Initializes the EventLogger.

        Args:
            log_file (str): Path to the log file.
        """
        self.log_file = log_file
        self.event_buffer = []

    def log_event(self, event_message, year):
        """
        Logs an event with a timestamp.

        Args:
            event_message (str): Description of the event.
            year (int): The simulation year when the event occurred.
        """
        self.event_buffer.append(f"[YEAR {year}] {event_message}")

    def flush_events(self):
        """
        Writes all buffered events to the log file.
        """
        if not self.event_buffer or self.log_file is None:
            return
        try:
            with open(self.log_file, "a") as f:
                for event in self.event_buffer:
                    f.write(event + "\n")
            self.event_buffer.clear()
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Failed to flush events to '{self.log_file}': {e}")

def merge_global_knowledge(local_knowledges):
    """
    Merges knowledge dictionaries from all processes into a single global knowledge base.

    Args:
        local_knowledges (list): List of local knowledge dictionaries.

    Returns:
        dict: Merged global knowledge dictionary.
    """
    merged = {}
    for kn in local_knowledges:
        merged.update(kn)
    return merged

def train_dqn(model, target_model, optimizer, replay_buffer):
    """
    Trains the DQN model using experiences from the replay buffer.

    Args:
        model (ActionModel): Current Q-network.
        target_model (ActionModel): Target Q-network.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        replay_buffer (ReplayBuffer): Buffer containing experiences.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    # Compute current Q values
    q_values = model(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        target_q = rewards + (1 - dones) * GAMMA * max_next_q

    # Compute loss and perform backpropagation
    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class Entity:
    """
    Represents an entity within the simulation world.
    """
    def __init__(self, name, role, health, location, resources=None, intelligence=0, alliances=None, skills=None, local_knowledge=None, home=None, town=None):
        """
        Initializes an Entity with various attributes.

        Args:
            name (str): Name of the entity.
            role (str): Role or occupation of the entity.
            health (int): Health level of the entity.
            location (tuple): (x, y) coordinates representing the entity's location.
            resources (dict, optional): Dictionary of resources. Defaults to None.
            intelligence (int, optional): Intelligence level. Defaults to 0.
            alliances (list, optional): List of alliances. Defaults to None.
            skills (list, optional): List of skills. Defaults to None.
            local_knowledge (set, optional): Set of local knowledge items. Defaults to None.
            home (str, optional): Name of the home location. Defaults to None.
            town (str, optional): Name of the town the entity belongs to. Defaults to None.
        """
        self.name = name
        self.role = role
        self.health = health
        self.location = location
        self.resources = resources or {"food": 100, "gold": 100, "influence": 50, "crafted_goods": 0}
        self.intelligence = intelligence
        self.alliances = alliances or []
        self.skills = skills or []
        self.local_knowledge = local_knowledge or set()
        self.last_state = None
        self.last_action = None
        self.home = home  # New attribute for home location
        self.town = town  # New attribute for town affiliation

    def get_state_vector(self, global_knowledge):
        """
        Constructs the state vector for the RL model based on the entity's attributes and global knowledge.

        Args:
            global_knowledge (dict): The collective knowledge discovered by entities.

        Returns:
            list: Normalized state vector.
        """
        state = [
            self.health / 100.0,
            self.intelligence / 100.0,
            self.resources["food"] / 1000.0,
            self.resources["gold"] / 1000.0,
            self.resources["influence"] / 1000.0,
            len(self.skills) / 100.0,
            float("advanced_agriculture" in global_knowledge),
            float("architecture" in global_knowledge),
            float("strategic_thinking" in global_knowledge),
            float("martial_arts" in global_knowledge),
            float("space_flight" in global_knowledge),      # New knowledge item
            float("town_management" in global_knowledge),  # New knowledge item
            float("home_building" in global_knowledge),    # New knowledge item
            float("trade_routes" in global_knowledge),
            float("alchemy" in global_knowledge)
        ]
        return state

    def possible_actions(self):
        """
        Lists all possible actions an entity can perform.

        Returns:
            list: List of action names.
        """
        return [
            "gather", "move", "rest", "plan", "build", "train", "explore",
            "experiment", "teach", "craft", "trade", "role_adapt",
            "create_item", "establish_home", "found_town", "initiate_space_flight"
        ]

    def select_action(self, global_knowledge, model, epsilon):
        """
        Selects an action based on the current policy (epsilon-greedy).

        Args:
            global_knowledge (dict): The collective knowledge.
            model (ActionModel or None): The RL model (only root rank uses it).
            epsilon (float): Current exploration rate.

        Returns:
            tuple: (chosen_action, state_vector, action_index)
        """
        actions = self.possible_actions()
        state_vec = self.get_state_vector(global_knowledge)

        # If model is None (non-root ranks), choose action randomly
        if model is None:
            action_index = random.randint(0, len(actions)-1)
            chosen_action = actions[action_index]
            return chosen_action, state_vec, action_index

        state_t = torch.tensor([state_vec], dtype=torch.float)
        if random.random() < epsilon:
            # Exploration: random action
            action_index = random.randint(0, len(actions)-1)
        else:
            # Exploitation: choose best action based on Q-values
            with torch.no_grad():
                q_values = model(state_t)
            action_index = q_values.argmax(dim=1).item()
        chosen_action = actions[action_index]
        return chosen_action, state_vec, action_index

    def can_discover_knowledge(self):
        """
        Determines if the entity can discover new knowledge based on role and intelligence.

        Returns:
            bool: True if the entity can discover knowledge, False otherwise.
        """
        base_chance = 0.0001
        if self.role == "Strategist":
            base_chance *= 5
        if self.role == "Leader":
            base_chance *= 2
        if self.intelligence > 50:
            base_chance *= 2
        return random.random() < base_chance

    def can_teach(self):
        """
        Determines if the entity can teach others based on intelligence and role.

        Returns:
            bool: True if the entity can teach, False otherwise.
        """
        return self.intelligence > 20 and self.role in ["Leader", "Strategist", "Educator"]

    def update(self, simulation_year, event_logger, global_knowledge, model, target_model, experience_container, epsilon):
        """
        Updates the entity's state by selecting and performing an action.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
            model (ActionModel or None): The RL model.
            target_model (ActionModel or None): The target RL model.
            experience_container (ReplayBuffer or list): Container for experiences.
            epsilon (float): Current exploration rate.
        """
        action, state_vec, action_index = self.select_action(global_knowledge, model, epsilon)
        self.last_state = state_vec
        self.last_action = action_index

        # Store old resources and intelligence to calculate reward
        old_resources_sum = sum(self.resources.values())
        old_intelligence = self.intelligence

        # Perform the chosen action
        self.perform_action(action, simulation_year, event_logger, global_knowledge)

        # Calculate reward based on changes
        new_resources_sum = sum(self.resources.values())
        reward = (new_resources_sum - old_resources_sum) * 0.01 + (self.intelligence - old_intelligence) * 0.1
        done = False  # Not used in this simulation
        next_state_vec = self.get_state_vector(global_knowledge)

        # Store experience
        exp_tuple = (self.last_state, self.last_action, reward, next_state_vec, done)
        if rank == 0:
            experience_container.push(exp_tuple)
        else:
            experience_container.append(exp_tuple)

    def perform_action(self, action, simulation_year, event_logger, global_knowledge):
        """
        Executes the specified action, modifying the entity's state accordingly.

        Args:
            action (str): The action to perform.
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if action == "gather":
            self.gather_resources(simulation_year, event_logger, global_knowledge)
        elif action == "move":
            self.move_entity(simulation_year, event_logger, global_knowledge)
        elif action == "rest":
            self.rest(simulation_year, event_logger, global_knowledge)
        elif action == "plan":
            self.plan_strategies(simulation_year, event_logger, global_knowledge)
        elif action == "build":
            self.build_structures(simulation_year, event_logger, global_knowledge)
        elif action == "train":
            self.train_skills(simulation_year, event_logger, global_knowledge)
        elif action == "explore":
            self.explore_land(simulation_year, event_logger, global_knowledge)
        elif action == "experiment":
            self.experiment_knowledge(simulation_year, event_logger, global_knowledge)
        elif action == "teach":
            self.teach_others(simulation_year, event_logger, global_knowledge)
        elif action == "craft":
            self.craft_goods(simulation_year, event_logger, global_knowledge)
        elif action == "trade":
            self.trade_goods(simulation_year, event_logger, global_knowledge)
        elif action == "role_adapt":
            self.adapt_role(global_knowledge)
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} adapted role to {self.role}.")
        elif action == "create_item":
            self.create_item(simulation_year, event_logger, global_knowledge)
        elif action == "establish_home":
            self.establish_home(simulation_year, event_logger, global_knowledge)
        elif action == "found_town":
            self.found_town(simulation_year, event_logger, global_knowledge)
        elif action == "initiate_space_flight":
            self.initiate_space_flight(simulation_year, event_logger, global_knowledge)

        # Handle random major events
        self.random_major_events(simulation_year, event_logger)

    def gather_resources(self, simulation_year, event_logger, global_knowledge):
        """
        Gathers food and gold resources.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        gather_bonus = 2 if "advanced_agriculture" in global_knowledge else 1
        self.resources["food"] += random.randint(5, 15) * gather_bonus
        self.resources["gold"] += random.randint(3, 8)
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} gathered resources.")

    def move_entity(self, simulation_year, event_logger, global_knowledge):
        """
        Moves the entity to a new location.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        # Entities move randomly within a bounded area
        new_x = self.location[0] + random.randint(-5, 5)
        new_y = self.location[1] + random.randint(-5, 5)
        self.location = (new_x, new_y)
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} moved to {self.location}.")

    def rest(self, simulation_year, event_logger, global_knowledge):
        """
        Restores the entity's health.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        rest_bonus = 2 if "herbal_medicine" in global_knowledge else 1
        self.health = min(100, self.health + random.randint(5, 15) * rest_bonus)
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} rested and recovered health.")

    def plan_strategies(self, simulation_year, event_logger, global_knowledge):
        """
        Increases the entity's intelligence through strategic planning.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        plan_bonus = 2 if "strategic_thinking" in global_knowledge else 1
        intelligence_gain = random.randint(3, 7) * plan_bonus
        self.intelligence += intelligence_gain
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} planned strategies, increasing intelligence by {intelligence_gain}.")

    def build_structures(self, simulation_year, event_logger, global_knowledge):
        """
        Builds structures, increasing influence.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        build_bonus = 2 if "architecture" in global_knowledge else 1
        influence_gain = random.randint(2, 6) * build_bonus
        self.resources["influence"] += influence_gain
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} built structures, increasing influence by {influence_gain}.")

    def train_skills(self, simulation_year, event_logger, global_knowledge):
        """
        Trains and acquires new skills.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        new_skill = f"Skill_{random.randint(1, 200)}"
        if new_skill not in self.skills:
            self.skills.append(new_skill)
            if "martial_arts" in global_knowledge and random.random() < 0.15:
                self.skills.append("Advanced_Combat")
            if "education" in global_knowledge and random.random() < 0.1:
                self.skills.append("Teaching")
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} trained and acquired new skills: {new_skill}.")

    def explore_land(self, simulation_year, event_logger, global_knowledge):
        """
        Explores new lands, potentially contributing to global knowledge.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        event_logger.log_event(f"{self.name} explored new lands.", simulation_year)
        if random.random() < 0.001:
            global_knowledge["geographic_survey"] = "Detailed knowledge of terrain"
            event_logger.log_event(f"{self.name} contributed 'geographic_survey' to global knowledge!", simulation_year)

    def experiment_knowledge(self, simulation_year, event_logger, global_knowledge):
        """
        Conducts experiments to discover new knowledge.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if self.can_discover_knowledge():
            discovered = self.discover_new_knowledge(global_knowledge)
            if discovered:
                event_logger.log_event(f"{self.name} discovered new knowledge: {discovered}", simulation_year)

    def teach_others(self, simulation_year, event_logger, global_knowledge):
        """
        Teaches others about known knowledge or skills.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        share_candidates = list(self.local_knowledge) + self.skills
        if not share_candidates:
            return
        shared = random.choice(share_candidates)
        event_logger.log_event(f"{self.name} held a teaching session on {shared}.", simulation_year)
        self.resources["influence"] += 2  # Increased influence gain for teaching
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} taught others about {shared}.")

    def discover_new_knowledge(self, global_knowledge):
        """
        Discovers a new piece of knowledge and updates global knowledge.

        Args:
            global_knowledge (dict): The collective knowledge.

        Returns:
            str or None: The key of the discovered knowledge or None if no discovery.
        """
        discoveries = [
            ("advanced_agriculture", "Improved farming techniques"),
            ("herbal_medicine", "Medicinal herbs and remedies"),
            ("strategic_thinking", "Better planning and warfare strategies"),
            ("architecture", "Building advanced structures and infrastructure"),
            ("martial_arts", "Advanced combat training methods"),
            ("smithing", "Craft advanced metal goods"),
            ("alchemy", "Transform resources into valuable compounds"),
            ("trade_routes", "Establish long-distance trade"),
            ("education", "Structured learning and teaching methods"),
            ("space_flight", "Technology to travel beyond the planet"),
            ("town_management", "Efficient management of towns and cities"),
            ("home_building", "Constructing sustainable homes")
        ]
        new_options = [d for d in discoveries if d[0] not in global_knowledge]
        if not new_options:
            return None
        discovered = random.choice(new_options)
        global_knowledge[discovered[0]] = discovered[1]
        self.local_knowledge.add(discovered[0])
        self.intelligence += 5
        if rank == 0:
            print(f"[INFO] {self.name} has discovered '{discovered[0]}': {discovered[1]}")
        return discovered[0]

    def random_major_events(self, simulation_year, event_logger):
        """
        Introduces random major events that can significantly affect the simulation.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
        """
        # Example major events
        if self.role == "Leader" and random.random() < 0.0005:
            event_logger.log_event(f"{self.name} founded a new city at {self.location}.", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} founded a new city at {self.location}.")
        if random.random() < 0.0003:
            event_logger.log_event(f"Natural disaster occurred near {self.location}, causing severe damage!", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] Natural disaster occurred near {self.location}, causing severe damage!")
        if random.random() < 0.0002:
            event_logger.log_event(f"{self.name} declared war on a neighboring faction!", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} declared war on a neighboring faction!")
        if random.random() < 0.00025:
            event_logger.log_event(f"{self.name} established a trade route, boosting economy.", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} established a trade route, boosting economy.")

    def create_item(self, simulation_year, event_logger, global_knowledge):
        """
        Creates a new item and adds it to the global knowledge.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        # Example items that can be created
        possible_items = [
            "Advanced_Sword", "Shield", "Potion", "Advanced_Technology", "Spacecraft", "Smart_Home"
        ]
        if self.resources["gold"] >= 50 and self.resources["food"] >= 20:
            new_item = random.choice(possible_items)
            self.resources["gold"] -= 50
            self.resources["food"] -= 20
            global_knowledge[new_item.lower()] = f"Description of {new_item}"
            event_logger.log_event(f"{self.name} created a new item: {new_item}.", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} created a new item: {new_item}.")
        else:
            # Not enough resources to create an item
            pass

    def establish_home(self, simulation_year, event_logger, global_knowledge):
        """
        Establishes a home location for the entity.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if "home_building" in global_knowledge and not self.home:
            self.home = f"Home_{self.name}"
            event_logger.log_event(f"{self.name} established a home: {self.home} at {self.location}.", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} established a home: {self.home} at {self.location}.")

    def found_town(self, simulation_year, event_logger, global_knowledge):
        """
        Founds a new town.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if "town_management" in global_knowledge and not self.town:
            self.town = f"Town_{self.name}"
            event_logger.log_event(f"{self.name} founded a new town: {self.town} at {self.location}.", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} founded a new town: {self.town} at {self.location}.")

    def initiate_space_flight(self, simulation_year, event_logger, global_knowledge):
        """
        Initiates space flight, advancing global knowledge.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if "space_flight" in global_knowledge and random.random() < 0.01:
            event_logger.log_event(f"{self.name} initiated space flight, expanding to new frontiers!", simulation_year)
            if rank == 0:
                print(f"[YEAR {simulation_year}] {self.name} initiated space flight, expanding to new frontiers!")

    def adapt_role(self, global_knowledge):
        """
        Adapts the entity's role based on current attributes and global knowledge.

        Args:
            global_knowledge (dict): The collective knowledge.
        """
        previous_role = self.role
        if self.intelligence > 80 and "strategic_thinking" in global_knowledge:
            self.role = "Strategist"
        elif self.resources["influence"] > 200 and "architecture" in global_knowledge:
            self.role = "Leader"
        elif "Advanced_Combat" in self.skills and "martial_arts" in global_knowledge:
            self.role = "Warrior"
        elif "smithing" in global_knowledge and "architecture" in global_knowledge:
            self.role = "Builder"
        elif "education" in global_knowledge and "Teaching" in self.skills:
            self.role = "Educator"
        if previous_role != self.role and rank == 0:
            print(f"[INFO] {self.name} changed role from {previous_role} to {self.role}.")

    def craft_goods(self, simulation_year, event_logger, global_knowledge):
        """
        Crafts goods using available resources and knowledge.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if "smithing" in global_knowledge and self.resources["gold"] >= 30:
            self.resources["gold"] -= 30
            self.resources["crafted_goods"] += 3
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} crafted goods using smithing.")
        elif "alchemy" in global_knowledge and self.resources["food"] >= 50:
            self.resources["food"] -= 50
            self.resources["crafted_goods"] += 5
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} crafted goods using alchemy.")
        else:
            # Insufficient resources to craft
            pass

    def trade_goods(self, simulation_year, event_logger, global_knowledge):
        """
        Engages in trading goods, boosting the economy.

        Args:
            simulation_year (int): The current simulation year.
            event_logger (EventLogger): Logger for recording events.
            global_knowledge (dict): The collective knowledge.
        """
        if "trade_routes" in global_knowledge:
            self.resources["food"] += 10
            self.resources["gold"] += 10
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} engaged in trade, boosting economy.")
        else:
            self.resources["gold"] += 2
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} traded basic goods.")

def initialize_world(num_entities):
    """
    Initializes the simulation world with a specified number of entities.

    Args:
        num_entities (int): Number of entities to create.

    Returns:
        list: List of Entity instances.
    """
    world = []
    roles = ["Explorer", "Trader", "Warrior", "Leader", "Builder", "Strategist", "Educator"]
    for i in range(num_entities):
        entity = Entity(
            name=f"Entity_{i}",
            role=random.choice(roles),
            health=100,
            location=(random.randint(-100, 100), random.randint(-100, 100))
        )
        world.append(entity)
    return world

def save_world(world, filename="world_save.json"):
    """
    Saves the current state of the world to a JSON file.

    Args:
        world (list): List of Entity instances.
        filename (str, optional): Path to the save file. Defaults to "world_save.json".
    """
    try:
        with open(filename, "w") as f:
            json.dump([{
                "name": e.name,
                "role": e.role,
                "health": e.health,
                "location": e.location,
                "resources": e.resources,
                "intelligence": e.intelligence,
                "alliances": e.alliances,
                "skills": e.skills,
                "local_knowledge": list(e.local_knowledge),
                "home": e.home,
                "town": e.town
            } for e in world], f)
        if rank == 0:
            print(f"[INFO] World saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save world to '{filename}': {e}")

def load_world(filename="world_save.json"):
    """
    Loads the world state from a JSON file.

    Args:
        filename (str, optional): Path to the save file. Defaults to "world_save.json".

    Returns:
        list or None: List of Entity instances if successful, else None.
    """
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] World save file '{filename}' does not exist. Initializing new world.")
        return None
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            if rank == 0:
                print(f"[INFO] World loaded successfully from '{filename}'.")
            return [
                Entity(
                    name=d["name"],
                    role=d["role"],
                    health=d["health"],
                    location=tuple(d["location"]),
                    resources=d["resources"],
                    intelligence=d["intelligence"],
                    alliances=d["alliances"],
                    skills=d["skills"],
                    local_knowledge=set(d.get("local_knowledge", [])),
                    home=d.get("home"),
                    town=d.get("town")
                ) for d in data
            ]
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load world from '{filename}': {e}")
        return None

def save_global_knowledge(global_knowledge, filename="global_knowledge.json"):
    """
    Saves the global knowledge to a JSON file.

    Args:
        global_knowledge (dict): The collective knowledge.
        filename (str, optional): Path to the save file. Defaults to "global_knowledge.json".
    """
    try:
        with open(filename, "w") as f:
            json.dump(global_knowledge, f, indent=4)
        if rank == 0:
            print(f"[INFO] Global knowledge saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save global knowledge to '{filename}': {e}")

def load_global_knowledge(filename="global_knowledge.json"):
    """
    Loads the global knowledge from a JSON file.

    Args:
        filename (str, optional): Path to the save file. Defaults to "global_knowledge.json".

    Returns:
        dict: The collective knowledge if successful, else empty dict.
    """
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] Global knowledge file '{filename}' does not exist. Initializing empty global knowledge.")
        return {}
    try:
        with open(filename, "r") as f:
            global_knowledge = json.load(f)
        if rank == 0:
            print(f"[INFO] Global knowledge loaded successfully from '{filename}'.")
        return global_knowledge
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load global knowledge from '{filename}': {e}")
        return {}

def save_model(model, filename="model_weights.pt"):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (ActionModel): The RL model.
        filename (str, optional): Path to the save file. Defaults to "model_weights.pt".
    """
    try:
        torch.save({
            'version': model.version,
            'state_dict': model.state_dict()
        }, filename)
        if rank == 0:
            print(f"[INFO] Model weights saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save model weights to '{filename}': {e}")

def load_model(model, filename="model_weights.pt"):
    """
    Loads the model's state dictionary from a file with weights_only=True.

    Args:
        model (ActionModel): The RL model.
        filename (str, optional): Path to the save file. Defaults to "model_weights.pt".
    """
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] Model weights file '{filename}' does not exist. Initializing model with random weights.")
        return
    try:
        # Use weights_only=True to comply with future PyTorch versions
        checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
        if 'version' in checkpoint and checkpoint['version'] != model.version:
            if rank == 0:
                print(f"[WARNING] Model version mismatch: checkpoint version {checkpoint['version']} vs current model version {model.version}. Initializing with random weights.")
            return
        model.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            print(f"[INFO] Model weights loaded successfully from '{filename}'.")
    except TypeError as te:
        # Handle versions where weights_only is not available
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        if rank == 0:
            print(f"[WARNING] 'weights_only' parameter not supported. Loaded model weights without it.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load model weights from '{filename}': {e}")

def save_state(state, filename="state.json"):
    """
    Saves the simulation state to a JSON file.

    Args:
        state (dict): Dictionary containing simulation state.
        filename (str, optional): Path to the save file. Defaults to "state.json".
    """
    try:
        with open(filename, "w") as f:
            json.dump(state, f, indent=4)
        if rank == 0:
            print(f"[INFO] Simulation state saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save simulation state to '{filename}': {e}")

def load_state(filename="state.json"):
    """
    Loads the simulation state from a JSON file.

    Args:
        filename (str, optional): Path to the save file. Defaults to "state.json".

    Returns:
        dict: Simulation state if successful, else default state.
    """
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] Simulation state file '{filename}' does not exist. Starting from year 0.")
        return {"simulation_year": 0, "steps": 0, "epsilon": EPSILON_START}
    try:
        with open(filename, "r") as f:
            state = json.load(f)
        if rank == 0:
            print(f"[INFO] Simulation state loaded successfully from '{filename}'.")
        return state
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load simulation state from '{filename}': {e}")
        return {"simulation_year": 0, "steps": 0, "epsilon": EPSILON_START}

def run_simulation():
    """
    Main function to run the simulation.
    """
    if rank == 0:
        # Introductory Information
        print("==========================================")
        print("      Welcome to the Enhanced World Simulation")
        print("==========================================\n")
        print("This simulation models a dynamic world with multiple entities that interact, gather resources, discover knowledge, create items, establish homes and towns, and even initiate space flight.")
        print("Reinforcement Learning (DQN) is utilized to optimize entity actions based on their states and experiences.\n")
        print("Saved Files:")
        print("1. 'world_save.json' - Stores the current state of all entities in the world.")
        print("2. 'global_knowledge.json' - Contains the collective knowledge discovered by entities.")
        print("3. 'model_weights.pt' - Saves the trained DQN model weights for future simulations.")
        print("4. 'state.json' - Stores the current simulation state including simulation year, training steps, and epsilon.\n")

        # Prompt to Reset Simulation
        reset_choice = input("Do you want to reset the simulation? This will delete existing saved data. (yes/no) [no]: ").strip().lower()
        if reset_choice in ['yes', 'y']:
            try:
                # Remove existing save files
                for file in ["world_save.json", "global_knowledge.json", "model_weights.pt", "state.json"]:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"[INFO] '{file}' has been deleted.")
                print("[INFO] Simulation has been reset.")
                # Initialize new world and state after reset
                world = initialize_world(5000)  # Increased number of entities for a richer simulation
                global_knowledge = {}
                state = {"simulation_year": 0, "steps": 0, "epsilon": EPSILON_START}
            except Exception as e:
                print(f"[ERROR] Failed to reset simulation: {e}")
                # Attempt to continue without resetting
                world = load_world() or initialize_world(5000)
                global_knowledge = load_global_knowledge() or {}
                state = load_state()
        else:
            # Do not reset, load existing data
            world = load_world() or initialize_world(5000)
            global_knowledge = load_global_knowledge() or {}
            state = load_state()

        # Prompt the user for the number of simulation years to run
        try:
            user_input = input("Enter the number of years to run the simulation: ")
            additional_years = int(user_input)
            print(f"[INFO] Simulation will run for {additional_years} years.")
        except ValueError:
            print("[ERROR] Invalid input for simulation years. Please enter an integer.")
            additional_years = 1000  # Default value
            print(f"[INFO] Simulation will run for {additional_years} years.")

        # Initialize models and optimizer
        model = ActionModel()
        target_model = ActionModel()
        target_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=LR)
        replay_buffer = ReplayBuffer()

        # Load model checkpoint if exists
        load_model(model, "model_weights.pt")
        load_model(target_model, "model_weights.pt")

        # Set simulation start year and other state variables
        simulation_year_start = state.get("simulation_year", 0)
        steps = state.get("steps", 0)
        epsilon = state.get("epsilon", EPSILON_START)

        # Start timing the simulation
        start_time = time.time()
        event_logger = EventLogger()
    else:
        # Non-root ranks do not handle I/O
        world = None
        global_knowledge = None
        model = None
        target_model = None
        optimizer = None
        replay_buffer = None
        simulation_year_start = 0
        additional_years = 0
        steps = 0
        epsilon = EPSILON_START
        event_logger = EventLogger(log_file=None)

    # Broadcast world, knowledge, and simulation parameters to all ranks
    world = comm.bcast(world, root=0)
    global_knowledge = comm.bcast(global_knowledge, root=0)
    simulation_year_start = comm.bcast(simulation_year_start, root=0)
    additional_years = comm.bcast(additional_years, root=0)
    steps = comm.bcast(steps, root=0)
    epsilon = comm.bcast(epsilon, root=0)

    # Determine the simulation end year
    simulation_year_end = simulation_year_start + additional_years
    year_duration = 0.001  # Real-time seconds per simulation year

    # Containers for experiences (root uses ReplayBuffer, others use local lists)
    local_experiences = []

    # Simulation Loop
    for simulation_year in range(simulation_year_start + 1, simulation_year_end + 1):
        # Determine which entities are handled by this rank
        entities_handled = world[rank::size]
        experience_container = replay_buffer if rank == 0 else local_experiences

        # Update each entity
        for entity in entities_handled:
            entity.update(
                simulation_year,
                event_logger,
                global_knowledge,
                model if rank == 0 else None,
                target_model if rank == 0 else None,
                experience_container,
                epsilon if rank == 0 else EPSILON_START
            )

        # Gather all knowledge and experiences from all ranks
        all_knowledge = comm.gather(global_knowledge, root=0)
        all_exps = comm.gather(local_experiences, root=0)

        if rank == 0:
            # Merge global knowledge from all ranks
            global_knowledge = merge_global_knowledge(all_knowledge)
            # Merge experiences from non-root ranks into replay_buffer
            for exp_list in all_exps:
                for exp in exp_list:
                    replay_buffer.push(exp)
            # Train the DQN model
            train_dqn(model, target_model, optimizer, replay_buffer)
            steps += 1
            # Update target model periodically
            if steps % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"[INFO] Target model updated at step {steps}.")
            # Decay epsilon
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Broadcast updated global knowledge, steps, and epsilon to all ranks
        global_knowledge = comm.bcast(global_knowledge, root=0)
        steps = comm.bcast(steps, root=0)
        epsilon = comm.bcast(epsilon, root=0)

        # Clear local experiences
        local_experiences.clear()
        comm.Barrier()  # Synchronize all ranks

        if rank == 0:
            # Provide updates every 100 years or at the end of simulation
            if simulation_year % 100 == 0 or simulation_year == simulation_year_end:
                elapsed = time.time() - start_time
                # Calculate some statistics
                total_food = sum(e.resources["food"] for e in world)
                total_gold = sum(e.resources["gold"] for e in world)
                total_influence = sum(e.resources["influence"] for e in world)
                total_crafted = sum(e.resources["crafted_goods"] for e in world)
                avg_intelligence = sum(e.intelligence for e in world) / len(world)
                num_knowledge = len(global_knowledge)
                num_towns = sum(1 for e in world if e.town)
                num_homes = sum(1 for e in world if e.home)
                print(f"\n[UPDATE] Year {simulation_year} - Elapsed Time: {elapsed/60:.2f} minutes")
                print(f"          Total Food: {total_food}, Total Gold: {total_gold}, Total Influence: {total_influence}, Crafted Goods: {total_crafted}")
                print(f"          Average Intelligence: {avg_intelligence:.2f}, Global Knowledge: {num_knowledge} items")
                print(f"          Number of Towns: {num_towns}, Number of Homes: {num_homes}\n")
                # Flush event logs to file
                event_logger.flush_events()
            # Pause to simulate real-time progression
            time.sleep(year_duration)

    if rank == 0:
        # After simulation ends, save all data
        event_logger.flush_events()
        save_world(world)
        save_global_knowledge(global_knowledge)
        save_model(model, "model_weights.pt")
        # Save simulation state
        state = {
            "simulation_year": simulation_year_end,
            "steps": steps,
            "epsilon": epsilon
        }
        save_state(state)
        print("[INFO] Simulation complete.")
        total_time = time.time() - start_time
        print(f"[RESULT] Total simulation runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    run_simulation()
sysop@master:~$
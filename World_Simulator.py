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

# ================= RL and Model Definitions =================

class ActionModel(nn.Module):
    def __init__(self, input_dim=10, action_dim=12):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Hyperparameters for DQN
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
TARGET_UPDATE = 1000  # Steps between target network updates
EPSILON_START = 0.1
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01

class EventLogger:
    def __init__(self, log_file="event_log.txt"):
        self.log_file = log_file
        self.event_buffer = []

    def log_event(self, event_message, year):
        self.event_buffer.append(f"[YEAR {year}] {event_message}")

    def flush_events(self):
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
    merged = {}
    for kn in local_knowledges:
        merged.update(kn)
    return merged

def train_dqn(model, target_model, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    # Compute current Q
    q_values = model(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q
    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        target_q = rewards + (1 - dones)*GAMMA*max_next_q

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class Entity:
    def __init__(self, name, role, health, location, resources=None, intelligence=0, alliances=None, skills=None, local_knowledge=None):
        self.name = name
        self.role = role
        self.health = health
        self.location = location
        self.resources = resources or {"food": 0, "gold": 0, "influence": 0, "crafted_goods": 0}
        self.intelligence = intelligence
        self.alliances = alliances or []
        self.skills = skills or []
        self.local_knowledge = local_knowledge or set()
        self.last_state = None
        self.last_action = None

    def get_state_vector(self, global_knowledge):
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
                float("martial_arts" in global_knowledge)
        ]
        return state

    def possible_actions(self):
        return ["gather", "move", "rest", "plan", "build", "train", "explore", "experiment", "teach", "craft", "trade", "role_adapt"]

    def select_action(self, global_knowledge, model, epsilon):
        actions = self.possible_actions()
        state_vec = self.get_state_vector(global_knowledge)
        # If model is None (non-root ranks), choose action randomly
        if model is None:
            action_index = random.randint(0, len(actions)-1)
            chosen_action = actions[action_index]
            return chosen_action, state_vec, action_index

        state_t = torch.tensor([state_vec], dtype=torch.float)
        if random.random() < epsilon:
            action_index = random.randint(0, len(actions)-1)
        else:
            with torch.no_grad():
                q_values = model(state_t)
            action_index = q_values.argmax(dim=1).item()
        chosen_action = actions[action_index]
        return chosen_action, state_vec, action_index

    def can_discover_knowledge(self):
        base_chance = 0.0001
        if self.role == "Strategist":
            base_chance *= 5
        if self.role == "Leader":
            base_chance *= 2
        if self.intelligence > 50:
            base_chance *= 2
        return random.random() < base_chance

    def can_teach(self):
        return self.intelligence > 20 and self.role in ["Leader", "Strategist"]

    def update(self, simulation_year, event_logger, global_knowledge, model, target_model, experience_container, epsilon):
        action, state_vec, action_index = self.select_action(global_knowledge, model, epsilon)
        self.last_state = state_vec
        self.last_action = action_index

        old_resources_sum = sum(self.resources.values())
        old_intelligence = self.intelligence

        self.perform_action(action, simulation_year, event_logger, global_knowledge)

        new_resources_sum = sum(self.resources.values())
        reward = (new_resources_sum - old_resources_sum)*0.01 + (self.intelligence - old_intelligence)*0.1
        done = False
        next_state_vec = self.get_state_vector(global_knowledge)

        # Store experience
        exp_tuple = (self.last_state, self.last_action, reward, next_state_vec, done)
        if rank == 0:
            experience_container.push(exp_tuple)
        else:
            experience_container.append(exp_tuple)

    def perform_action(self, action, simulation_year, event_logger, global_knowledge):
        if action == "gather":
            gather_bonus = 2 if "advanced_agriculture" in global_knowledge else 1
            self.resources["food"] += random.randint(1, 10)*gather_bonus
            self.resources["gold"] += random.randint(1, 5)
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} gathered resources.")
        elif action == "move":
            self.location = (self.location[0] + random.randint(-1, 1),
                             self.location[1] + random.randint(-1, 1))
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} moved to {self.location}.")
        elif action == "rest":
            rest_bonus = 2 if "herbal_medicine" in global_knowledge else 1
            self.health = min(100, self.health + random.randint(1, 5)*rest_bonus)
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} rested and recovered health.")
        elif action == "plan":
            plan_bonus = 2 if "strategic_thinking" in global_knowledge else 1
            self.intelligence += random.randint(1, 3)*plan_bonus
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} planned strategies, increasing intelligence.")
        elif action == "build":
            build_bonus = 2 if "architecture" in global_knowledge else 1
            self.resources["influence"] += random.randint(1, 4)*build_bonus
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} built structures, increasing influence.")
        elif action == "train":
            new_skill = f"Skill_{random.randint(1, 100)}"
            if new_skill not in self.skills:
                self.skills.append(new_skill)
                if "martial_arts" in global_knowledge and random.random() < 0.1:
                    self.skills.append("Advanced_Combat")
                if rank == 0 and simulation_year % 100 == 0:
                    print(f"[YEAR {simulation_year}] {self.name} trained and acquired new skills: {new_skill}.")
        elif action == "explore":
            event_logger.log_event(f"{self.name} explored new lands.", simulation_year)
            if random.random() < 0.001:
                global_knowledge["geographic_survey"] = "Detailed knowledge of terrain"
                event_logger.log_event(f"{self.name} contributed geographic_survey to global knowledge!", simulation_year)
        elif action == "experiment":
            if self.can_discover_knowledge():
                discovered = self.discover_new_knowledge(global_knowledge)
                if discovered:
                    event_logger.log_event(f"{self.name} discovered new knowledge: {discovered}", simulation_year)
        elif action == "teach":
            if self.can_teach():
                self.teach_others(simulation_year, event_logger, global_knowledge)
        elif action == "craft":
            if "smithing" in global_knowledge and self.resources["gold"] > 10:
                self.resources["gold"] -= 10
                self.resources["crafted_goods"] += 1
                if rank == 0 and simulation_year % 100 == 0:
                    print(f"[YEAR {simulation_year}] {self.name} crafted goods using smithing.")
            elif "alchemy" in global_knowledge and self.resources["food"] > 20:
                self.resources["food"] -= 20
                self.resources["crafted_goods"] += 1
                if rank == 0 and simulation_year % 100 == 0:
                    print(f"[YEAR {simulation_year}] {self.name} crafted goods using alchemy.")
        elif action == "trade":
            if "trade_routes" in global_knowledge:
                self.resources["food"] += 5
                self.resources["gold"] += 5
                if rank == 0 and simulation_year % 100 == 0:
                    print(f"[YEAR {simulation_year}] {self.name} engaged in trade, boosting economy.")
            else:
                self.resources["gold"] += 1
                if rank == 0 and simulation_year % 100 == 0:
                    print(f"[YEAR {simulation_year}] {self.name} traded basic goods.")
        elif action == "role_adapt":
            self.adapt_role(global_knowledge)
            if rank == 0 and simulation_year % 100 == 0:
                print(f"[YEAR {simulation_year}] {self.name} adapted role to {self.role}.")

        self.random_major_events(simulation_year, event_logger)

    def adapt_role(self, global_knowledge):
        previous_role = self.role
        if self.intelligence > 60 and "strategic_thinking" in global_knowledge:
            self.role = "Strategist"
        elif self.resources["influence"] > 100 and "architecture" in global_knowledge:
            self.role = "Leader"
        elif "Advanced_Combat" in self.skills and "martial_arts" in global_knowledge:
            self.role = "Warrior"
        elif "smithing" in global_knowledge and "architecture" in global_knowledge:
            self.role = "Builder"
        if previous_role != self.role and rank == 0:
            print(f"[INFO] {self.name} changed role from {previous_role} to {self.role}.")

    def teach_others(self, simulation_year, event_logger, global_knowledge):
        share_candidates = list(self.local_knowledge) + self.skills
        if not share_candidates:
            return
        shared = random.choice(share_candidates)
        event_logger.log_event(f"{self.name} held a teaching session on {shared}.", simulation_year)
        self.resources["influence"] += 1
        if rank == 0 and simulation_year % 100 == 0:
            print(f"[YEAR {simulation_year}] {self.name} taught others about {shared}.")

    def discover_new_knowledge(self, global_knowledge):
        discoveries = [
            ("advanced_agriculture", "Improved farming techniques"),
            ("herbal_medicine", "Medicinal herbs and remedies"),
            ("strategic_thinking", "Better planning and warfare strategies"),
            ("architecture", "Building advanced structures and infrastructure"),
            ("martial_arts", "Advanced combat training methods"),
            ("smithing", "Craft advanced metal goods"),
            ("alchemy", "Transform resources into valuable compounds"),
            ("trade_routes", "Establish long-distance trade")
        ]
        new_options = [d for d in discoveries if d[0] not in global_knowledge]
        if not new_options:
            return None
        discovered = random.choice(new_options)
        global_knowledge[discovered[0]] = discovered[1]
        self.local_knowledge.add(discovered[0])
        self.intelligence += 5
        if rank == 0:
            print(f"[INFO] {self.name} has discovered {discovered[0]}: {discovered[1]}")
        return discovered[0]

    def random_major_events(self, simulation_year, event_logger):
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

def initialize_world(num_entities):
    world = []
    roles = ["Explorer", "Trader", "Warrior", "Leader", "Builder", "Strategist"]
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
                "local_knowledge": list(e.local_knowledge)
            } for e in world], f)
        if rank == 0:
            print(f"[INFO] World saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save world to '{filename}': {e}")

def load_world(filename="world_save.json"):
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] World save file '{filename}' does not exist. Initializing new world.")
        return None
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            if rank == 0:
                print(f"[INFO] World loaded successfully from '{filename}'.")
            return [Entity(**{k:v for k,v in d.items() if k in ["name","role","health","location","resources","intelligence","alliances","skills"]},
                           local_knowledge=set(d.get("local_knowledge", []))) for d in data]
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load world from '{filename}': {e}")
        return None

def save_state(state, filename="state.json"):
    try:
        with open(filename, "w") as f:
            json.dump(state, f)
        if rank == 0:
            print(f"[INFO] Simulation state saved successfully to '{filename}'.")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to save simulation state to '{filename}': {e}")

def load_state(filename="state.json"):
    if not os.path.exists(filename):
        if rank == 0:
            print(f"[INFO] Simulation state file '{filename}' does not exist. Starting from year 1.")
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
    if rank == 0:
        # Introductory Information
        print("==========================================")
        print("      Welcome to the World Simulation")
        print("==========================================")
        print("\nThis simulation models a world with multiple entities that interact, gather resources, discover knowledge, and evolve over time.")
        print("Reinforcement Learning (DQN) is used to optimize entity actions based on their states and experiences.\n")
        print("Saved Files:")
        print("1. 'world_save.json' - Stores the current state of all entities in the world.")
        print("2. 'global_knowledge.json' - Contains the collective knowledge discovered by entities.")
        print("3. 'model_weights.pt' - Saves the trained DQN model weights for future simulations.")
        print("4. 'state.json' - Stores the current simulation state including simulation year, training steps, and epsilon.\n")

        # Prompt to Reset Simulation
        reset_choice = input("Do you want to reset the simulation? This will delete existing saved data. (yes/no) [no]: ").strip().lower()
        if reset_choice in ['yes', 'y']:
            try:
                if os.path.exists("world_save.json"):
                    os.remove("world_save.json")
                    print("[INFO] 'world_save.json' has been deleted.")
                if os.path.exists("global_knowledge.json"):
                    os.remove("global_knowledge.json")
                    print("[INFO] 'global_knowledge.json' has been deleted.")
                if os.path.exists("model_weights.pt"):
                    os.remove("model_weights.pt")
                    print("[INFO] 'model_weights.pt' has been deleted.")
                if os.path.exists("state.json"):
                    os.remove("state.json")
                    print("[INFO] 'state.json' has been deleted.")
                print("[INFO] Simulation has been reset.")
                # Initialize new world and state after reset
                world = initialize_world(5000)
                global_knowledge = {}
                state = {"simulation_year": 0, "steps": 0, "epsilon": EPSILON_START}
            except Exception as e:
                print(f"[ERROR] Failed to reset simulation: {e}")
                # Attempt to continue without resetting
                world = load_world() or initialize_world(5000)
                global_knowledge = load_world("global_knowledge.json") or {}
                state = load_state()
        else:
            # Do not reset, load existing data
            world = load_world() or initialize_world(5000)
            # Load global knowledge
            if os.path.exists("global_knowledge.json"):
                try:
                    with open("global_knowledge.json","r") as f:
                        global_knowledge = json.load(f)
                    print("[INFO] Global knowledge loaded successfully from 'global_knowledge.json'.")
                except Exception as e:
                    print(f"[ERROR] Failed to load global knowledge from 'global_knowledge.json': {e}")
                    global_knowledge = {}
            else:
                global_knowledge = {}
                print("[INFO] No existing global knowledge found. Initializing empty global knowledge.")
            # Load simulation state
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
        if os.path.exists("model_weights.pt"):
            try:
                # Attempt to load with weights_only=True
                model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu'), weights_only=True))
                target_model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu'), weights_only=True))
                print("[INFO] Model weights loaded successfully from 'model_weights.pt'.")
            except TypeError:
                # If weights_only is not supported, load normally
                model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu')))
                target_model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu')))
                print("[WARNING] 'weights_only' parameter not supported. Loaded model weights without it.")
            except Exception as e:
                print(f"[ERROR] Failed to load model weights from 'model_weights.pt': {e}")
        else:
            print("[INFO] No existing model weights found. Initializing model with random weights.")

        # Set simulation start year and other state variables
        simulation_year_start = state.get("simulation_year", 0)
        steps = state.get("steps", 0)
        epsilon = state.get("epsilon", EPSILON_START)

        start_time = time.time()
        event_logger = EventLogger()
    else:
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

    # Broadcast world, knowledge, and simulation_year_start, additional_years, steps, epsilon
    world = comm.bcast(world, root=0)
    global_knowledge = comm.bcast(global_knowledge, root=0)
    simulation_year_start = comm.bcast(simulation_year_start, root=0)
    additional_years = comm.bcast(additional_years, root=0)
    steps = comm.bcast(steps, root=0)
    epsilon = comm.bcast(epsilon, root=0)

    # Determine the simulation range
    simulation_year_end = simulation_year_start + additional_years
    year_duration = 0.001  # Real-time seconds per simulation year
    local_experiences = []

    for simulation_year in range(simulation_year_start + 1, simulation_year_end + 1):
        experience_container = replay_buffer if rank == 0 else local_experiences
        for entity in world[rank::size]:
            entity.update(simulation_year, event_logger, global_knowledge, model if rank==0 else None, target_model, experience_container, epsilon if rank==0 else EPSILON_START)

        # Gather all knowledge and experiences
        local_knowledge_data = global_knowledge
        all_knowledge = comm.gather(local_knowledge_data, root=0)
        all_exps = comm.gather(local_experiences, root=0)

        if rank == 0:
            # Merge global knowledge
            global_knowledge = merge_global_knowledge(all_knowledge)
            # Merge experiences from non-root ranks into replay_buffer
            for exp_list in all_exps:
                for exp in exp_list:
                    replay_buffer.push(exp)
            # Train DQN
            train_dqn(model, target_model, optimizer, replay_buffer)
            steps += 1
            if steps % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"[INFO] Target model updated at step {steps}.")
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Broadcast updated global knowledge, steps, and epsilon
        global_knowledge = comm.bcast(global_knowledge, root=0)
        steps = comm.bcast(steps, root=0)
        epsilon = comm.bcast(epsilon, root=0)

        # Clear local experiences
        local_experiences.clear()
        comm.Barrier()

        if rank == 0:
            # Provide more frequent updates, e.g., every 100 years
            if simulation_year % 100 == 0 or simulation_year == simulation_year_end:
                elapsed = time.time() - start_time
                # Calculate some statistics
                total_food = sum(e.resources["food"] for e in world)
                total_gold = sum(e.resources["gold"] for e in world)
                total_influence = sum(e.resources["influence"] for e in world)
                total_crafted = sum(e.resources["crafted_goods"] for e in world)
                avg_intelligence = sum(e.intelligence for e in world) / len(world)
                num_knowledge = len(global_knowledge)
                print(f"[UPDATE] Year {simulation_year} - Elapsed Time: {elapsed/60:.2f} minutes")
                print(f"          Total Food: {total_food}, Total Gold: {total_gold}, Total Influence: {total_influence}, Crafted Goods: {total_crafted}")
                print(f"          Average Intelligence: {avg_intelligence:.2f}, Global Knowledge: {num_knowledge} items")
                event_logger.flush_events()
            time.sleep(year_duration)

    if rank == 0:
        # After simulation ends, update and save all data
        event_logger.flush_events()
        save_world(world)
        try:
            with open("global_knowledge.json","w") as f:
                json.dump(global_knowledge, f)
            print("[INFO] Global knowledge saved successfully to 'global_knowledge.json'.")
        except Exception as e:
            print(f"[ERROR] Failed to save global knowledge to 'global_knowledge.json': {e}")
        try:
            torch.save(model.state_dict(), "model_weights.pt")
            print("[INFO] Model weights saved successfully to 'model_weights.pt'.")
        except Exception as e:
            print(f"[ERROR] Failed to save model weights to 'model_weights.pt': {e}")
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
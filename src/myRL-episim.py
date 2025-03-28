import numpy as np
import subprocess
from datetime import datetime, timedelta
import os
import json
import xarray
from epi_sim import EpiSim

def run_episim():
    "Run steps and update the policy"
    executable_path = os.path.join(os.pardir, "episim")

    initial_conditions = os.path.join(os.pardir, "models/mitma/initial_conditions.nc")

    # read the config file sample to dict
    with open(os.path.join(os.pardir, "models/mitma/config.json"), 'r') as f:
        config = json.load(f)

    data_folder = os.path.join(os.pardir, "models/mitma")
    instance_folder = os.path.join(os.pardir, "runs")

    model = EpiSim(
        config, data_folder, instance_folder, initial_conditions
    )
    
    # Set up with compiled executable
    # model.setup(executable_type='compiled', executable_path=os.path.join(os.pardir, "episim"))
    
    # Or set up with Julia interpreter
    model.setup(executable_type='interpreter', executable_path=os.path.join(os.pardir, "run.jl"))

    logger.debug("debug Model wrapper init complete")

    start_date="2023-01-01"
    logger.info(f"First date: {start_date}")
    current_date = start_date
    for i in range(1):
        new_state, next_date = model.step(start_date=current_date, length_days=7)
        # new_state, next_date = model.step(current_date, 7)

        # update the policy
        # increase the level of lockdown by 5% at each iteration
        config["NPI"]["Îºâ‚€s"] = [ config["NPI"]["Îºâ‚€s"][0] * (1 - 0.05) ]
        model.update_config(config)

        logger.debug(f"Iteration {i+1} - Model state: {new_state}")
        logger.info(f"Iteration {i+1} - Next date: {next_date}")
        current_date = next_date

    logger.info("Example done")





# Environment Interface
class CustomEnv:
    def __init__(self):
        # Define environment state and action space
        # Episode duration: 1 year (48 weeks)
        # Step: 2 weeks
        self.state_dims = (48, 125, 5, 5, 5, 2)
        # self.state_space = 6  # State is a vector of size six [weeks(1-48), previous_actions(1-125), ICU_stress(1-5), disease_spread(1-5), dis_severity(1-5), R0(0/1)]
        self.action_space = 125  # 125 possible actions [\Phi0(0,0.25,0.5,0.75,1), delta(0,0.25,0.5,0.75,1), k0(0,0.25,0.5,0.75,1)]
        self.state = None

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns:
            state (numpy array): The initial state.
        """
        self.state =  tuple(np.random.randint(dim) for dim in self.state_dims) #TODO: run simulator and get INIT state
        return self.state

    def step(self, action):
        """
        Applies the given action to the environment.
        Args:
            action (int): The chosen action.
        Returns:
            state (numpy array): The next state. 
            reward (float): The reward obtained.
            done (bool): Whether the episode is finished.
        """
        # Simulate environment dynamics
        # Invoke the simulator:
        # subprocess.call(['python3', 'src/epi_sim.py'])
        # determine week no.

        # Convert action to the corresponding parameters in the .json file

        # Invoke the simulator with that .json file

        # Read the output and proceed

        util = Utils()
        cf = util.get_most_recent_folder(os.path.join(os.pardir,'runs'))
        print(f"ID of current exp: {cf}")
        f = open(os.path.join(os.pardir,f"runs/{cf}/config_auto_py.json"))
        temp_conf = json.load(f)
        week_state = util.get_week_number(temp_conf['simulation']['start_date'])
        print(f"Week no: {week_state}")
    # HERE 17-12-2024
        # subprocess.call(['python3', 'src/epi_sim.py'])
    subprocess.call(['./episim', '-e', 'MMCACovid19', 'run', '-c', 'models/mitma/config.json', '-d', 'models/mitma/', '--initial-condition', 'models/mitma/initial_conditions_MMCACovid19.nc'])
        #run_episim()

        self.state = tuple(np.random.randint(dim) for dim in self.state_dims) #TODO: run simulator and get NEXT state
        reward = np.random.randn()  # Example: Random reward #TODO: run simulator and get reward
        done = np.random.rand() > 0.95  # Example: Randomly ends the episode #TODO: run simulator and get determine if it is week 48
        return self.state, reward, done

    def render(self):
        """
        Renders the current state of the environment.
        """
        print(f"State: {self.state}")

# Step 2: RL Agent
class RLAgent:
    def __init__(self, state_dims, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initializes the Q-Learning Agent.
        Args:
            state_dims (list): Number of discrete values for each state dimension.
            action_space (int): Number of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial epsilon for exploration.
            epsilon_decay (float): Decay rate of epsilon per episode.
            min_epsilon (float): Minimum value of epsilon.
        """
        self.state_dims = state_dims
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table initialization
        # self.state_bins = [np.linspace(0, 1, bins) for _ in range(state_space)]
        self.q_table = np.zeros(state_dims + (action_space,))

    def select_action(self, state):
        """
        Selects an action using the epsilon-greedy policy.
        Args:
            state (numpy array): Current state.
        Returns:
            action (int): Chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_space)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Temporal Difference (TD) method.
        Args:
            state (numpy array): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy array): Next state.
            done (bool): Whether the episode ended.
        """
        # discretized_state = self.discretize_state(state)
        # discretized_next_state = self.discretize_state(next_state)

        # TD Target
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + self.gamma * max_next_q

        # TD Update
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon).
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Step 3: Training Loop
def train_agent(env, agent, episodes=100):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")


class Utils:
    def get_week_number(self, date_str):
        """
        Determines the week number (1-48) for a given date in 2023.
        
        Args:
            date_str (str): The date in "YYYY-MM-DD" format.
        
        Returns:
            int: The week number (1-48).
        """
        # Ensure the date is in 2020
        year_start = datetime(2020, 1, 1)
        year_end = datetime(2020, 12, 31)

        # Parse the input date
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        if not (year_start <= date <= year_end):
            raise ValueError("Date is out of range. Provide a date between 2020-01-01 and 2020-12-31.")
        
        # Calculate the difference in days from the start of the year
        day_difference = (date - year_start).days
        
        # Determine the week number (1-based index)
        week_number = day_difference // 7 + 1
        
        if week_number > 48:
            raise ValueError("The date exceeds the 48th week of 2020.")
        
        return week_number

    def get_most_recent_folder(self, directory):
        """
        Finds the most recently modified folder in the specified directory.

        Args:
            directory (str): The path to the directory to search.

        Returns:
            str: The name of the most recent folder, or None if no folders are found.
        """
        try:
            # List all entries in the directory
            entries = os.listdir(directory)

            # Filter only folders
            folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

            if not folders:
                print("No folders found in the directory.")
                return None

            # Get the most recently modified folder
            most_recent_folder = max(folders, key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))
            return most_recent_folder
        except FileNotFoundError:
            print(f"The directory '{directory}' does not exist.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# Initialize and run
if __name__ == "__main__":
    env = CustomEnv()
    agent = RLAgent(state_dims=env.state_dims, action_space=env.action_space)
    train_agent(env, agent, episodes=1000)
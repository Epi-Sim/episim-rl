import numpy as np
import subprocess
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import xarray as xr
import shutil
import argparse
import math
import pickle


def log_episode_reward(output_path, episode_number, total_reward):
    """
    Logs the total reward of an episode to a CSV file.
    Creates the file if it doesn't exist.

    Args:
        output_path (str): Path to the CSV file to save rewards.
        episode_number (int): The episode number.
        total_reward (float): The total reward obtained.
    """
    log_exists = os.path.exists(output_path)
    with open(output_path, 'a') as f:
        if not log_exists:
            f.write("episode,total_reward\n")  # Header
        f.write(f"{episode_number},{total_reward}\n")

def log_reward(output_path, episode_number, week, total_hosp, new_hosp, new_deaths, H_term, D_term, lockdown_term, reward, action, delta, k0, error):
    """
    Logs in the detail the reward function of an episode to a CSV file.
    Creates the file if it doesn't exist.

    Args:
        output_path (str): Path to the CSV file to save metrics.
        episode_number (int): The episode number.
        week (int): The current week number.
        Total_hosp (float): Total hospitalizations.
        new_hosp (float): New hospitalizations. 
        new_deaths (float): New deaths.
        H_term (float): Hospitalization term.
        D_term (float): Death term.
        Lockdown_term (float): Lockdown term.
        reward (float): The reward obtained.
        delta (float): Change in action.
        action (int): The chosen action.
        k0 (float): Parameter k0.
        phi (float): Parameter phi.
    """
    log_exists = os.path.exists(output_path)
    with open(output_path, 'a') as f:
        if not log_exists:
            f.write("episode,week,Total_hosp,new_hosp,new_deaths,H_term,D_term,Lockdown_term,reward,action,delta,k0,error\n")  # Header
        f.write(f"{episode_number},{week},{total_hosp},{new_hosp},{new_deaths},{H_term},{D_term},{lockdown_term},{reward},{action},{delta},{k0},{error}\n")


#function that maps values of the observables to the state space 1-5 or 0-1
def map_observables_to_state_space(value, map_dict):
    thresholds = {int(k): v for k, v in map_dict.items()}  # Convert keys to integers
    sorted_thresholds = sorted(thresholds.items())  # Sort thresholds
    
    new_value = 1  # Default to the lowest category
    for threshold, category in sorted_thresholds:
        if value >= threshold:
            new_value = category
        else:
            break
    return new_value
    

def map_to_action(data_folder, action):
    fname = os.path.join(data_folder, "map_action.csv")
    df = pd.read_csv(fname, index_col='action')
    return df.loc[action]

#create a function that creates the action space id and map it to the possible actions
#action is the id




# Environment Interface
class CustomEnv:
    def __init__(self, base_folder, run_folder, data_folder, config_dict, categories_dict, evaluation_period, episode_length, config_file):
        # Define environment state and action space
        # Episode duration: 1 year (48 weeks)
        # Step: 2 weeks
        self.base_folder = base_folder
        self.run_folder = run_folder
        self.data_folder = data_folder
        self.config_file = config_file
        self.config_dict = config_dict
        self.categories_dict = categories_dict
        self.evaluation_period = evaluation_period
        self.state_dims = (48, 121, 5, 5, 5, 2)
        self.state_space = 6  # State is a vector of size six [weeks(1-48), previous_actions(0-120), ICU_stress(0-4), disease_spread(0-4), dis_severity(0-4), R0(0/1)]
        self.action_space = 121  # 121 possible actions [delta(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), k0(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)]
        self.state = None
        self.steps = 0
        self.episode_length = episode_length
        

    def reset(self, episode):
        """
        Resets the environment to the initial state.
        Returns:
            state (numpy array): The initial state.
        """

        print(f"Resetting")

        self.steps = 0

        initial_conditions = ["initial_conditions-good.nc", "initial_conditions-med.nc", "initial_conditions-bad.nc"]

        utils = Utils()        

        config_file_template = os.path.join(self.run_folder, "config_template.json")

        with open(config_file_template, 'r') as f:
            config_dict_up = json.load(f)

        new_start_date = config_dict_up['simulation']['start_date']
        new_end_date = (datetime.strptime(new_start_date, "%Y-%m-%d") + timedelta(days=self.evaluation_period)).strftime("%Y-%m-%d")
        self.config_dict['simulation']['start_date'] = new_start_date
        self.config_dict['simulation']['end_date'] = new_end_date

        self.config_dict["NPI"]["κ₀s"]= [0] * (self.evaluation_period + 1)
        self.config_dict["NPI"]["δs"]= [0] * (self.evaluation_period + 1)
        phi = config_dict_up["NPI"]["ϕs"]
        self.config_dict["NPI"]["ϕs"]= phi * (self.evaluation_period + 1)
        self.config_dict["NPI"]["tᶜs"]= list(range(1, self.evaluation_period + 2))

        rand = np.random.randint(0, len(initial_conditions))
        self.config_dict['data']['initial_condition_filename'] = initial_conditions[rand]

        week_state = utils.get_week_number(self.config_dict['simulation']['start_date']) - 1
        print(f"Resetting ... week no: {week_state}")
        self.state = (week_state, 0, 0, 0, 0, 0)

        return self.state

    def step(self, action, episode):
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
        utils = Utils()

        week_state = utils.get_week_number(self.config_dict['simulation']['start_date']) - 1
        print(f"Week no: {week_state}")

        if int(week_state) >= self.episode_length - 1:
            done = True
            return self.state, 0, done
        # HERE 17-12-2024
        # subprocess.call(['python3', 'src/epi_sim.py'])


        # Read the population data
        population_fname = os.path.join(self.data_folder, self.config_dict['data']['metapopulation_data_filename'])
        population = pd.read_csv(population_fname, index_col = 'id', usecols = ["id", "Y", "M", "O"])
        total_population = population[['Y', 'M', 'O']].sum().sum()

        if self.steps == 0:
            initial_conditions = self.config_dict['data']['initial_condition_filename']
            initial_observables = xr.open_dataset(os.path.join(self.data_folder, initial_conditions))

            ICU_stress = (float(initial_observables["HR"].sum(["G","M"]).values) +
                          float(initial_observables["HD"].sum(["G","M"]).values))
            disease_spread = float(initial_observables["I"].sum(['G','M']).values) * 100000 / total_population
            dis_severity = float(initial_observables["D"].sum(["G", "M"]).values)

            new_hospitalizations = ICU_stress

            kappa = float(self.config_dict["NPI"]["κ₀s"][0])
            delta = float(self.config_dict["NPI"]["δs"][0])
            A = 25
            b = 0.2
            a = 1.5
            lockdown_cost = A*(1/(1 + math.exp(-a*(kappa - b)))) + (1/(1 + math.exp(-a*(delta - b))))

            total_hosp = ICU_stress
            new_deaths = dis_severity

            ICU_stress = map_observables_to_state_space(ICU_stress, categories_dict['ICU_stress'])
            disease_spread = map_observables_to_state_space(disease_spread, categories_dict['disease_spread'])
            dis_severity = map_observables_to_state_space(dis_severity, categories_dict['dis_severity'])
            R0 = 1

            # Update the state
            self.state = (week_state, action, ICU_stress, disease_spread, dis_severity, R0)

            # COMPUTE REWARD
            # Lockdown term
            lockdown_term = (-0.5 * (ICU_stress**2 + dis_severity**2) + A)*lockdown_cost
            # Hospitalization term
            H_term = new_hospitalizations*(ICU_stress + 1)

            #REWARD 
            reward = -(H_term + new_deaths + lockdown_term)
            print(f"Initial step reward: {reward}")
            done = False
            self.steps = self.steps + 1

            return self.state, reward, done


        
        # Convert action to the corresponding parameters in the .json file
        #APPLY ACTION
        #TODO I think that in the first two weeks no action has to be made
        #if week_state > 6:
        if self.steps > 0:
            action_values = map_to_action(data_folder, action)
            self.config_dict["NPI"]["κ₀s"]= [float(action_values['k0'])] * self.evaluation_period
            self.config_dict["NPI"]["δs"]= [float(action_values['delta'])] * self.evaluation_period
            print(f"**-- selected action: {action}, maps to: {action_values}")
       # else:
            #TODO: What values is supposed to have action in the first two weeks?
            #action = np.random.randint(125)
		
        # Invoke the simulator with that .json file

        config_fname = os.path.join(self.run_folder, f"config_{episode}_{week_state}.json")
        with open(config_fname, "w") as fh:
            json.dump(self.config_dict, fh, indent=4)

        params_strn = f"-c {config_fname} -d {self.data_folder} -i {self.run_folder}"
            
        command = f"julia {exec_path} run {params_strn}"
        subprocess.run(command, shell=True)
 
        last_day = self.config_dict['simulation']['end_date']

        # Read the output and proceed
        
        

        # read the output observables and compute the reward

        full_xa = xr.open_dataset(os.path.join(self.run_folder, "output", "compartments_full.nc"))
        observables_xa = xr.open_dataset(os.path.join(self.run_folder, "output", "observables.nc"))

        ICU_stress = float(full_xa["HR"].sel(T=last_day).sum(['G','M']).values) + float(full_xa["HD"].sel(T=last_day).sum(['G','M']).values)
        disease_spread = float(full_xa["I"].sel(T=last_day).sum(['G','M']).values) * 100000 / total_population
        dis_severity = float(observables_xa["new_deaths"].sum(['G','M','T']).values)
        R0_xa = observables_xa["R_eff"].sel(T=last_day) * population / total_population
        R0 = float(R0_xa.sum(['G', 'M']).values)


        # Check for errors in the simulation
        # If the simulation fails, re-run it
        error = False

        if (ICU_stress > 10**10 or dis_severity > 10**10):
            error = True
            print("Simulation failed. Re-running simulation")
            subprocess.run(command, shell=True)

            full_xa = xr.open_dataset(os.path.join(self.run_folder, "output", "compartments_full.nc"))
            observables_xa = xr.open_dataset(os.path.join(self.run_folder, "output", "observables.nc"))

            ICU_stress = float(full_xa["HR"].sel(T=last_day).sum(['G','M']).values) + float(full_xa["HD"].sel(T=last_day).sum(['G','M']).values)
            disease_spread = float(full_xa["I"].sel(T=last_day).sum(['G','M']).values) * 100000 / total_population
            dis_severity = float(observables_xa["new_deaths"].sum(['G','M','T']).values)
            R0_xa = observables_xa["R_eff"].sel(T=last_day) * population / total_population
            R0 = float(R0_xa.sum(['G', 'M']).values)
    

        # Calculate new hospitalizations
        new_hospitalizations = float(observables_xa["new_hospitalized"].sum(['G','M','T']).values)
    
        # Calculate the lockdown cost based on the NPI parameters
        kappa = float(self.config_dict["NPI"]["κ₀s"][0])
        delta = float(self.config_dict["NPI"]["δs"][0])
        A = 25
        b = 0.2
        a = 1.5
        lockdown_cost = A*(1/(1 + math.exp(-a*(kappa - b)))) + (1/(1 + math.exp(-a*(delta - b))))
        
        total_hosp = ICU_stress
        new_deaths = dis_severity        
        
        #MAP OBSERVABLES TO STATE SPACE
        ICU_stress = map_observables_to_state_space(ICU_stress, categories_dict['ICU_stress'])
        disease_spread = map_observables_to_state_space(disease_spread, categories_dict['disease_spread'])
        dis_severity = map_observables_to_state_space(dis_severity, categories_dict['dis_severity'])
        R0 = map_observables_to_state_space(R0, categories_dict['R0'])
        
        # Update the state
        self.state = (week_state, action, ICU_stress, disease_spread, dis_severity, R0)

        # COMPUTE REWARD
        # Lockdown term
        lockdown_term = (-0.5 * (ICU_stress**2 + dis_severity**2) + A)*lockdown_cost
        # Hospitalization term
        H_term = new_hospitalizations*(ICU_stress + 1)

        #REWARD 
        reward = -(H_term + new_deaths + lockdown_term)


        # Save reward function and the contribution of each term (H_term, new_deaths, lockdown_term) to a CSV file
        output_path = os.path.join(self.run_folder, "reward_contributions.csv")

        log_reward(output_path, episode, week_state, total_hosp, new_hospitalizations, new_deaths, H_term, new_deaths, lockdown_term, reward, action, delta, kappa, error)

        #self.state = tuple(np.random.randint(dim) for dim in self.state_dims) #TODO: run simulator and get NEXT state

        #TODO store each value to make a plot of the reward
        
        #done has to be true when week 48
        #done = np.random.rand() > 0.95  # Example: Randomly ends the episode #TODO: run simulator and get determine if it is week 48
        done = False

        new_start_day = self.config_dict['simulation']['end_date']
        self.config_dict['simulation']['start_date'] = new_start_day
        new_end_date = (datetime.strptime(new_start_day, "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
        self.config_dict['simulation']['end_date'] = new_end_date

        initial_condition_filename = os.path.join(self.base_folder, self.run_folder, "output", f"compartments_t_{new_start_day}.nc")
        self.config_dict['data']['initial_condition_filename'] = initial_condition_filename

        self.steps = self.steps + 1
 
        #cf = util.get_most_recent_folder(os.path.join("","test"))
        #print(f"ID of current exp: {cf}")
        #f = open(os.path.join(os.pardir,f"runs/{cf}/config_auto_py.json"))

        return self.state, reward, done

    def render(self, episode):
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

    def learn(self, state, action, reward, next_state, run_folder, done):
        """
        Updates the Q-table using the Temporal Difference (TD) method.
        Args:
            state (numpy array): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy array): Next state.
            run_folder (str): Folder to save the Q-table.
            done (bool): Whether the episode ended.
        """
        # discretized_state = self.discretize_state(state)
        # discretized_next_state = self.discretize_state(next_state)

        # TD Target
        try:
            max_next_q = np.max(self.q_table[next_state]) if not done else 0
        except IndexError:
            max_next_q = 0
        td_target = reward + self.gamma * max_next_q

        # TD Update
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        
        with open(os.path.join(run_folder, "q_table.pkl"), "wb") as f:
            pickle.dump(self.q_table, f)

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon).
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Step 3: Training Loop
def train_agent(env, agent, episodes=2):
    rewards_log_file = os.path.join(env.run_folder, "episode_rewards.csv")

    for episode in range(episodes):
        state = env.reset(episode)
        total_reward = 0
        done = False

        while not done:
            env.render(episode)
            action = agent.select_action(state)
            next_state, reward, done = env.step(action, episode)
            print(f"**-- Selected action: {action}, Next state: {next_state}, Reward: {reward}")
            agent.learn(state, action, reward, next_state, env.run_folder, done)
            state = next_state
            total_reward += reward
            print(f"**** Episode {episode + 1} (Step {env.steps}): Reward = {reward:.2f}, Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
            
        agent.decay_epsilon()
        print(f"**** Episode {episode + 1}: Total Reward = {total_reward:.5f}, Epsilon = {agent.epsilon:.3f}")

        log_episode_reward(rewards_log_file, episode + 1, total_reward)

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




def create_parser():
    parser = argparse.ArgumentParser(description=f"Run the EpiSim simulator.")
    parser.add_argument("--experiment_id", action="store", dest="experiment_id", help="ID of the experiment")
    parser.add_argument("--config", action="store", required=True, dest="config_file", help="Path to the configuration file")
    parser.add_argument("--data", action="store", required=True, dest="data_folder", help="Folder where the data is stored")
    parser.add_argument("--period", action="store", dest="evaluation_period", help="Evaluation period", type=int, default=14)
    parser.add_argument("--episodes", action="store", dest="episodes", help="Number of episodes to run", type=int, default=10)
    parser.add_argument("--episode_length", action="store", dest="episode_length", help="Episode length", type=int, default=48)
    return parser


# Initialize and run

if __name__ == "__main__":
    global exec_path

    parser = create_parser()
    args = parser.parse_args()

    base_folder = os.path.abspath(os.curdir)
    exec_path = os.path.join(base_folder, "model/EpiSim.jl/src/run.jl")

    experiment_id = args.experiment_id
    data_folder = args.data_folder
    config_file = args.config_file
    evaluation_period = args.evaluation_period
    episodes = args.episodes
    episode_length = args.episode_length

    assert evaluation_period > 0, "The evaluation period must be a positive integer."
    assert evaluation_period <= 336, "The evaluation period must be less than or equal to 48 weeks."
    assert os.path.exists(config_file), "The configuration file does not exist."
    assert os.path.exists(data_folder), "The data folder does not exist."


    exp_folder = os.path.join("runs", experiment_id)
    #Delete the folder if it exists
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    # Create the experiment folder
    os.makedirs(exp_folder, exist_ok=True)

    #Copy the data folder to the experiment folder
    data_exp_folder = os.path.join(exp_folder, "data")
    shutil.copytree(data_folder, data_exp_folder)
    data_folder = data_exp_folder

    config_file_template = os.path.join(exp_folder, "config_template.json")
    shutil.copy(config_file, config_file_template)
    config_file = config_file_template


    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    categorization_fname = os.path.join(data_folder,"observables_categories.json")
    with open(categorization_fname, "r") as f:
            categories_dict = json.load(f)

    

    env = CustomEnv(base_folder=base_folder, run_folder=exp_folder, data_folder=data_folder, config_dict=config_dict, categories_dict=categories_dict, evaluation_period=evaluation_period, episode_length=episode_length, config_file=config_file)
    agent = RLAgent(state_dims=env.state_dims, action_space=env.action_space)
    train_agent(env, agent, episodes=episodes)

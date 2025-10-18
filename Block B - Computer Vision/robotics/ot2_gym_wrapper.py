import gymnasium as gym
from gymnasium import spaces
import numpy as np
# We assume the simulation class is in a file named sim_class.py
# If the file has a different name, you'll need to change the import.
from sim_class import Simulation

class OT2Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, max_steps=1000):
        super(OT2Env, self).__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Create the simulation environment
        # The render argument in Simulation might need to be adjusted
        # based on the implementation of the Simulation class.
        self.sim = Simulation(num_agents=1, render=(self.render_mode == 'human'))

        # --- Define Action and Observation Spaces ---
        # They must be gym.spaces objects

        # Action space: 3D vector for velocity control [x, y, z]
        # Values are normalized between -1 and 1.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation space: 6D vector containing:
        # [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        # We use -inf and +inf as bounds since the exact working envelope
        # is managed by the simulation.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Keep track of the number of steps
        self.steps = 0
        self.goal_position = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent within the working envelope.
        low_bounds = np.array([-0.2, -0.2, 0.1])
        high_bounds = np.array([0.2, 0.2, 0.4])
        self.goal_position = self.np_random.uniform(low=low_bounds, high=high_bounds)

        # Call the environment reset function
        sim_obs = self.sim.reset(num_agents=1)
        
        # **FIX**: Dynamically get the robot ID and extract the pipette position.
        robot_id = next(iter(sim_obs))
        pipette_pos = np.array(sim_obs[robot_id]['pipette_position'], dtype=np.float32)

        # Concatenate pipette position and goal position for the final observation
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def step(self, action):
        # **FIX**: The simulation expects a 4D action vector. Append 0 for the drop action.
        full_action = np.concatenate([action, [0]], dtype=np.float32)
        
        # The simulation expects a list of actions, one for each agent.
        sim_obs = self.sim.run([full_action])

        # **FIX**: Dynamically get the robot ID and extract the pipette position.
        robot_id = next(iter(sim_obs))
        pipette_pos = np.array(sim_obs[robot_id]['pipette_position'], dtype=np.float32)
        
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # --- Calculate the reward ---
        distance_to_goal = np.linalg.norm(pipette_pos - self.goal_position)
        reward = -distance_to_goal

        # --- Check for termination condition ---
        terminated = False
        success_threshold = 0.01
        if distance_to_goal < success_threshold:
            terminated = True
            reward += 100

        # --- Check for truncation condition ---
        self.steps += 1
        truncated = self.steps >= self.max_steps

        # Info dictionary (can be used for debugging)
        info = {'distance_to_goal': distance_to_goal}

        return observation, reward, terminated, truncated, info

    def render(self):
        pass # not necessary here

    def close(self):

        self.sim.close()

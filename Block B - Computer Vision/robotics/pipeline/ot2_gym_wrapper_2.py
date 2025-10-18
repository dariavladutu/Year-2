import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    """
    This is the advanced environment wrapper for the OT-2 simulation.
    It includes customizable reward parameters for more effective training.
    """
    def __init__(self, render=False, max_steps=1000, threshold=0.001, 
                 # **MODIFIED**: Default values are now tuned for high accuracy.
                 bonus_reward=150,
                 reward_distance_scale=200, 
                 step_penalty=-1):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.threshold = threshold
        self.bonus_reward = bonus_reward
        self.reward_distance_scale = reward_distance_scale
        self.step_penalty = step_penalty

        # Create the simulation environment
        # **FIX**: Pass the 'render' flag to the Simulation class to control visualization.
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-0.1874, -0.1711, 0.1195, -0.1874, -0.1711, 0.1195]),
            high=np.array([0.253, 0.2202, 0.2902, 0.253, 0.2202, 0.2902]),
            shape=(6,),
            dtype=np.float32
        )
        self.steps = 0
        self.goal_position = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set a random goal position for the agent
        self.goal_position = self.np_random.uniform(
            low=[-0.1874, -0.1711, 0.1195], high=[0.253, 0.2202, 0.2902]
        )

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Get the correct robot ID dynamically
        robot_key = list(observation.keys())[0]
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)

        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)
        self.steps = 0

        return observation, {}
        
    def step(self, action):
        # The original file from your friend had a scaled action and a 4th element.
        # This is the correct implementation based on that file.
        scaled_action = np.append(action * 0.5, 0)

        # Call the environment step function
        observation = self.sim.run([scaled_action])

        # Get the correct robot ID dynamically
        robot_key = list(observation.keys())[0]
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)

        # Ensure pipette stays within the working envelope
        pipette_position = np.clip(pipette_position, self.observation_space.low[:3], self.observation_space.high[:3])
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # --- This is the correct, advanced reward calculation ---
        distance = np.linalg.norm(pipette_position - self.goal_position)
        # The reward uses the parameters from __init__
        reward = -self.reward_distance_scale * distance + self.step_penalty

        # Add bonus reward for success
        terminated = bool(distance < self.threshold)
        if terminated:
            reward += self.bonus_reward

        truncated = bool(self.steps >= self.max_steps)
        info = {"success": terminated}
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()

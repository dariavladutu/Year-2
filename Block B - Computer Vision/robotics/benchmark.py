import numpy as np
import time
import csv
from datetime import datetime
import os

# Import both the environment and the controllers
from ot2_gym_wrapper_2 import OT2Env 
from pid_controller import PIDController
from stable_baselines3 import PPO

def save_benchmark_results(filename, data):
    """Saves the results of a benchmark run to a CSV file."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def run_benchmark(controller_type, model_or_pid, test_suite, num_trials, accuracy_threshold_mm=1.0):
    """
    Runs a full benchmark for a given controller.

    Args:
        controller_type (str): 'PID' or 'RL'.
        model_or_pid: The loaded RL model or an instance of the PIDController.
        test_suite (list): A list of 3D target coordinates to test.
        num_trials (int): The number of times to run the test for each target.
        accuracy_threshold_mm (float): The success threshold in millimeters.
    """
    print(f"\n--- Starting Benchmark for {controller_type} Controller ---")
    
    env = OT2Env(render=False) # No rendering for faster benchmarking
    accuracy_threshold_m = accuracy_threshold_mm / 1000.0

    for target_pos in test_suite:
        print(f"\nTesting Target: {target_pos}")
        for trial in range(num_trials):
            # --- Reset and Setup ---
            obs, _ = env.reset()
            env.goal_position = np.array(target_pos)
            if controller_type == 'RL':
                obs[3:] = env.goal_position
            elif controller_type == 'PID':
                model_or_pid.set_target(env.goal_position)

            # --- Metrics Initialization ---
            settling_time = -1
            max_overshoot = 0
            total_distance_traveled = 0
            last_position = obs[:3]
            
            # --- Simulation Loop ---
            for step in range(env.max_steps):
                current_position = obs[:3]
                
                # Update metrics
                total_distance_traveled += np.linalg.norm(current_position - last_position)
                last_position = current_position
                
                error = np.linalg.norm(env.goal_position - current_position)
                if error > max_overshoot:
                    max_overshoot = error

                # Check for settling time
                if error <= accuracy_threshold_m and settling_time == -1:
                    settling_time = step + 1

                # Get action from the appropriate controller
                if controller_type == 'RL':
                    action, _ = model_or_pid.predict(obs, deterministic=True)
                else: # PID
                    action = model_or_pid.update(current_position)
                
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            
            # --- Log Results for this Trial ---
            final_error_m = np.linalg.norm(env.goal_position - obs[:3])
            
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "controller_type": controller_type,
                "trial_num": trial + 1,
                "target_x": target_pos[0],
                "target_y": target_pos[1],
                "target_z": target_pos[2],
                "final_error_mm": final_error_m * 1000,
                "settling_time_steps": settling_time,
                "max_overshoot_mm": max_overshoot * 1000,
                "path_efficiency_m": total_distance_traveled
            }
            save_benchmark_results("benchmark_results.csv", log_data)
            print(f"  Trial {trial + 1}: Error={final_error_m*1000:.2f}mm, Settling Time={settling_time} steps")

    env.close()

def main():
    """Main function to run the full benchmark."""
    
    # --- Configuration ---
    # Define the set of target positions for the benchmark.
    TEST_SUITE = [
        [0.1, 0.15, 0.25],    # A standard target
        [-0.1, -0.1, 0.15],   # A target in a different quadrant
        [0.2, 0.0, 0.28],     # A target near the edge
        [0.0, 0.0, 0.2],      # A target in the center
    ]
    NUM_TRIALS = 5 # Number of runs for each target position
    
    # --- PID Controller Benchmark ---
    pid_gains = {'kp': 5.0, 'ki': 0.5, 'kd': 2.0}
    pid_controller = PIDController(**pid_gains)
    run_benchmark('PID', pid_controller, TEST_SUITE, NUM_TRIALS)

    # --- RL Agent Benchmark ---
    # **IMPORTANT**: Change this to the path of your best-trained RL model.
    RL_MODEL_PATH = "models/0jfld8sq/final_model.zip" # Example path
    try:
        rl_model = PPO.load(RL_MODEL_PATH, device="cpu")
        run_benchmark('RL', rl_model, TEST_SUITE, NUM_TRIALS)
    except FileNotFoundError:
        print(f"\nCould not find RL model at {RL_MODEL_PATH}. Skipping RL benchmark.")
    except Exception as e:
        print(f"\nAn error occurred while loading or running the RL model: {e}")

    print("\n--- Benchmark Complete ---")
    print(f"Results have been saved to {os.path.abspath('benchmark_result.csv')}")

if __name__ == '__main__':
    main()

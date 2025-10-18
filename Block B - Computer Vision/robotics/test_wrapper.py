import gymnasium as gym
from ot2_gym_wrapper import OT2Env
from stable_baselines3.common.env_checker import check_env
import time

def main():
    print("--- Starting Environment Test ---")

    # Instantiate the custom environment
    try:
        env = OT2Env(render_mode='human')
    except NameError:
        print("\nError: Could not find the 'Simulation' class.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred while creating the environment: {e}")
        return


    # --- Step 1: Check environment compatibility ---
    # This utility from Stable Baselines 3 will check if your environment
    # follows the gymnasium API. If it doesn't, it will raise an error.
    print("\n[1] Checking environment compatibility with Stable Baselines 3...")
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        env.close()
        return

    # --- Step 2: Run a simple test with random actions ---
    print("\n[2] Running test with random actions for 1000 steps...")

    try:
        # Reset the environment to get the initial observation
        obs, info = env.reset()
        total_reward = 0

        for step in range(1000):
            # Take a random action from the environment's action space
            random_action = env.action_space.sample()

            # Execute the action in the environment
            obs, reward, terminated, truncated, info = env.step(random_action)

            total_reward += reward
            
            if (step + 1) % 100 == 0:
                 print(f"Step: {step + 1}, Action: {random_action}, Reward: {reward:.4f}")

            # If the episode is terminated or truncated, reset the environment
            if terminated or truncated:
                print("\nEpisode finished.")
                if terminated:
                    print("Reason: Goal was reached (Terminated).")
                if truncated:
                    print("Reason: Max steps reached (Truncated).")
                
                print(f"Total reward for the episode: {total_reward:.4f}")
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0


            # A small delay to make the visualization easier to follow
            time.sleep(1./240.) # Sleep for a time consistent with PyBullet's default physics simulation step

    except Exception as e:
        print(f"\nAn error occurred during the test loop: {e}")
    finally:
        # Close the environment
        print("\n--- Test Finished ---")
        env.close()

if __name__ == '__main__':
    main()

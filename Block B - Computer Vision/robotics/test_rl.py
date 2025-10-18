import numpy as np
from stable_baselines3 import PPO # Use the same algorithm you trained with
import time

# Use the environment wrapper from your friend's code
from ot2_gym_wrapper import OT2Env 

def main():
    """
    Main function to test a trained RL agent in the OT-2 environment.
    """
    print("--- Starting RL Agent Test ---")

    # --- Configuration ---
    MODEL_PATH = "models/0jfld8sq/final_model.zip" 

    # Define the target position you want the pipette to move to.
    TARGET_POSITION = np.array([0.1, 0.15, 0.25])

    # Number of simulation steps to run the test for.
    TEST_STEPS = 1000

    # --- Initialization ---
    try:
        # Load the trained model.
        model = PPO.load(MODEL_PATH, device='cpu')
        print(f"Model loaded from {MODEL_PATH}")

        # Initialize the environment with rendering.
        env = OT2Env(render=True)
    
    except TypeError:
        # Fallback for visualization if 'render' argument is not accepted.
        print("Note: 'render=True' not accepted. Trying without it for visualization.")
        env = OT2Env()


    except Exception as e:
        print(f"\nAn error occurred during initialization: {e}")
        return

    # Reset the environment.
    obs, _ = env.reset()
    
    # Manually set the goal for testing purposes.
    env.goal_position = TARGET_POSITION
    
    # The observation needs to be updated to reflect the new goal
    obs[3:] = TARGET_POSITION
    
    current_position = obs[:3]

    print(f"Target Position: {TARGET_POSITION}")
    print(f"Initial Position: {current_position}")
    print("\nRunning simulation with trained agent...")

    # --- Simulation Loop ---
    try:
        for step in range(TEST_STEPS):
            # Get the action from the trained model.
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute the action in the environment.
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update the current position.
            current_position = obs[:3]
            
            # A small delay for visualization.
            time.sleep(1./240.)

            if terminated:
                print("\nAgent reached the target!")
                break
        
        if not terminated and not truncated:
            print("\nTest finished before agent reached the target.")
        elif truncated:
            print("\nEpisode truncated (max steps reached).")


    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during the test loop: {e}")
    finally:
        # --- Results ---
        print("\n--- Test Finished ---")
        final_position = current_position
        final_error_m = np.linalg.norm(TARGET_POSITION - final_position)
        final_error_mm = final_error_m * 1000

        print(f"Final Position: {final_position}")
        print(f"Final Error (Distance to Target): {final_error_m:.6f} m ({final_error_mm:.3f} mm)")

        # Check if the controller meets the accuracy requirements.
        accuracy_req_10mm = 0.01
        accuracy_req_1mm = 0.001
        status = ""

        if final_error_m <= accuracy_req_1mm:
            status = f"Success: Met the highest accuracy requirement of {accuracy_req_1mm * 1000:.0f} mm."
            print(f"✅  {status}")
        elif final_error_m <= accuracy_req_10mm:
            status = f"Success: Met the base accuracy requirement of {accuracy_req_10mm * 1000:.0f} mm."
            print(f"✅  {status}")
        else:
            status = f"Failure: Did not meet the accuracy requirement of {accuracy_req_10mm * 1000:.0f} mm."
            print(f"❌  {status}")
            
        env.close()

if __name__ == '__main__':
    main()

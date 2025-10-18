import numpy as np
from ot2_gym_wrapper import OT2Env
from pid_controller import PIDController
import time
import csv
import os

def save_results_to_csv(filename, data):
    """
    Saves the results of a PID test run to a CSV file.

    Args:
        filename (str): The name of the CSV file to save to.
        data (dict): A dictionary containing the data to be saved.
    """
    # Check if the file exists to determine if we need to write headers.
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write headers only if the file is new.
            
        writer.writerow(data)
        
    # Print the absolute path to make the file easy to find.
    absolute_path = os.path.abspath(filename)
    print(f"\nResults saved to: {absolute_path}")

def main():
    """
    Main function to test the PID controller with the OT-2 environment.
    """
    print("--- Starting PID Controller Test ---")

    # --- Configuration ---
    # Define the target position you want the pipette to move to.
    TARGET_POSITION = np.array([0.1, 0.15, 0.25])

    KP = 5.0
    KI = 0.5
    KD = 2.0

    # Number of simulation steps to run the test for.
    TEST_STEPS = 500

    # --- Initialization ---
    try:
        # Initialize the environment. render_mode='human' to see the simulation.
        env = OT2Env(render_mode='human')
        
        # Initialize the PID controller with the chosen gains.
        pid = PIDController(kp=KP, ki=KI, kd=KD)
    except Exception as e:
        print(f"\nAn unexpected error occurred during initialization: {e}")
        return

    # Set the target for the PID controller.
    pid.set_target(TARGET_POSITION)

    # Reset the environment to get the initial observation.
    obs, _ = env.reset()
    current_position = obs[:3] # The first 3 values are the pipette position

    print(f"Target Position: {TARGET_POSITION}")
    print(f"Initial Position: {current_position}")
    print(f"PID Gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print("\nRunning simulation...")

    # --- Simulation Loop ---
    try:
        for step in range(TEST_STEPS):
            # Calculate the control action using the PID controller.
            action = pid.update(current_position)
            
            # Execute the action in the environment.
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update the current position from the new observation.
            current_position = obs[:3]
            
            # A small delay to make the visualization easier to follow.
            time.sleep(1./240.)

            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{TEST_STEPS}...")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during the test loop: {e}")
    finally:
        # --- Results ---
        print("\n--- Test Finished ---")
        final_position = current_position
        final_error_m = np.linalg.norm(TARGET_POSITION - final_position)
        final_error_mm = final_error_m * 1000 # Convert meters to millimeters

        # **FIX**: Correctly print both meters and millimeters.
        print(f"Final Error (Distance to Target): {final_error_m:.6f} m ({final_error_mm:.3f} mm)")

        # **FIX**: Use clearer variable names for requirements.
        accuracy_req_10mm = 0.01  # 10 mm
        accuracy_req_1mm = 0.001 # 1 mm
        status = ""

        # **FIX**: Correct the logical order to check the strictest requirement first.
        if final_error_m <= accuracy_req_1mm:
            status = f"Success: Met the highest accuracy requirement of {accuracy_req_1mm * 1000:.0f} mm."
            print(f"{status}")
        elif final_error_m <= accuracy_req_10mm:
            status = f"Success: Met the base accuracy requirement of {accuracy_req_10mm * 1000:.0f} mm."
            print(f"{status}")
        else:
            status = f"Failure: Did not meet the accuracy requirement of {accuracy_req_10mm * 1000:.0f} mm."
            print(f"{status}")
        
        # --- Logging ---
        log_data = {
            "kp": KP,
            "ki": KI,
            "kd": KD,
            "test_steps": TEST_STEPS,
            "target_x": TARGET_POSITION[0],
            "target_y": TARGET_POSITION[1],
            "target_z": TARGET_POSITION[2],
            "final_x": final_position[0],
            "final_y": final_position[1],
            "final_z": final_position[2],
            "error_m": f"{final_error_m:.6f}",
            "error_mm": f"{final_error_mm:.3f}",
            "status": status
        }
        save_results_to_csv("pid_tuning_log.csv", log_data)

        # Close the environment.
        env.close()

if __name__ == '__main__':
    main()

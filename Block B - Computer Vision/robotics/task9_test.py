from sim_class import Simulation
import time

# --- Setup ---
print("Initializing Simulation...")
sim = Simulation(num_agents=1)
print("Simulation Initialized.")

# --- The Challenge: Find a corner ---

# 1. Define an action to move the robot towards its maximum X, Y, and Z limits.
# We set high velocities to ensure it moves quickly to the corner.
actions = [[1.0, 1.0, 1.0, 0]] # [Max vel_x, Max vel_y, Max vel_z, no-drop]

# 2. We need to store the previous position to know when the robot stops moving.
last_position = None
# This loop will run as long as the robot is still moving.
while True:
    # Send the command for a single step
    state = sim.run(actions)
    
    # Get the current pipette position
    current_position = state['robotId_1']['pipette_position']
    
    # Check if the robot has stopped moving
    # If the current position is the same as the last position, it has hit a limit.
    if current_position == last_position:
        print("\nRobot has stopped moving. Corner found!")
        break # Exit the loop
        
    # Update the last known position
    last_position = current_position
    
    # Print the current position in a clean format to see the progress
    print(f"Current Position: X={current_position[0]:.4f}, Y={current_position[1]:.4f}, Z={current_position[2]:.4f}", end='\r')

# 3. Print the final coordinates of the corner
print("\n" + "="*50)
print(f"Recorded Corner (+X, +Y, +Z): {last_position}")
print("="*50)

# Keep the window open to observe the final position
time.sleep(10)
print("Script finished.")
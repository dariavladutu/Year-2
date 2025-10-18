import time
import numpy as np
from sim_class import Simulation

# --- Configuration ---
SPEED = 0.5
PAUSE_DURATION = 0.1  # Seconds to pause at each corner.
TOLERANCE = 0.01  # How close the robot needs to get to the corner to be "done".

# --- Path Definition ---
# These are the 8 corner coordinates we found in the task9_test_2.
# We've arranged them into a logical path for the tour.
path_coordinates = [
    [-0.1872, -0.1706, 0.1195],  # Start: bottom-left-front
    [ 0.253,  -0.1707, 0.1195],  # Move to bottom-right-front
    [ 0.2533,  0.2195, 0.1195],  # Move to bottom-right-back
    [-0.1872,  0.2195, 0.1196],  # Move to bottom-left-back
    [-0.1872,  0.2195, 0.2895],  # Move up to top-left-back
    [ 0.2533,  0.2195, 0.2896],  # Move to top-right-back
    [ 0.2532, -0.1706, 0.2895],  # Move to top-right-front
    [-0.1872, -0.1706, 0.2895],  # Move to top-left-front
]

# --- Simulation ---
print("Initializing Simulation...")
sim = Simulation(num_agents=1)
print("Starting Robot Tour...")

# Loop through each target position in our defined path
for i, target_pos in enumerate(path_coordinates):
    print(f"Moving to corner {i+1}/{len(path_coordinates)}: {target_pos}")
    
    while True:
        # Get the robot's current state and position
        state = sim.run([[0,0,0,0]]) # Send a no-op action to get state
        robot_key = list(state.keys())[0]
        current_pos = np.array(state[robot_key]['pipette_position'])
        
        # Calculate the distance and direction to the target
        direction_vector = target_pos - current_pos
        distance = np.linalg.norm(direction_vector)
        
        # Check if we have arrived at the corner
        if distance < TOLERANCE:
            print(f"Arrived at corner {i+1}.")
            time.sleep(PAUSE_DURATION) # Pause at the corner
            break # Exit the while loop to move to the next corner
            
        # If not there yet, calculate velocity and send command
        # Normalize the direction vector and multiply by speed to get velocity
        normalized_direction = direction_vector / distance
        velocity = normalized_direction * SPEED
        
        # Send the movement command
        action = [list(velocity) + [0]]
        sim.run(action)

print("\nTour complete!")
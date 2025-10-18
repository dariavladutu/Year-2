import csv
from sim_class import Simulation

def move_until_stopped(sim, velocity_vector, max_duration=500):
    """Applies a velocity until the robot stops moving and returns the final state."""
    action = [velocity_vector]
    last_pos = None
    
    for _ in range(max_duration):
        state = sim.run(action)
        
        # Dynamically get the robot's ID key
        robot_key = list(state.keys())[0]
        current_pos = state[robot_key]['pipette_position']
        
        # If the position is the same as the last step, we've stopped.
        if current_pos == last_pos:
            return state
            
        last_pos = current_pos
        
    # Failsafe in case it never stops
    print("Warning: Max duration reached in move_until_stopped.")
    return state

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Define the REVISED sequence of velocities
    movement_sequence = [
        # --- Bottom Layer ---
        [-1.0, -1.0, -1.0, 0], # 1. Go to bottom-left-front
        [ 1.0,  0.0,  0.0, 0], # 2. Slide to bottom-right-front
        [ 0.0,  1.0,  0.0, 0], # 3. Slide to bottom-right-back
        [-1.0,  0.0,  0.0, 0], # 4. Slide to bottom-left-back

        # --- Move to Top Layer ---
        # This is the key change. We move up from the last known corner.
        [ 0.0,  0.0,  1.0, 0], # 5. Move straight up to top-left-back

        # --- Top Layer ---
        [ 1.0,  0.0,  0.0, 0], # 6. Slide to top-right-back
        [ 0.0, -1.0,  0.0, 0], # 7. Slide to top-right-front
        [-1.0,  0.0,  0.0, 0], # 8. Slide to top-left-front
    ]


    # 2. Initialize simulation and a list to hold results
    sim = Simulation(num_agents=1)
    recorded_path = []

    print("Starting sequential movement...")

    # 3. Loop through the movement sequence
    for i, velocity in enumerate(movement_sequence):
        print(f"Executing move {i+1}/{len(movement_sequence)}...")
        
        # Call our function to move the robot
        final_state = move_until_stopped(sim, velocity)
        
        # Get the final position and save it
        robot_key = list(final_state.keys())[0]
        final_position = final_state[robot_key]['pipette_position']
        recorded_path.append(final_position)
        print(f"  > Stopped at: {final_position}")

    # 4. Save the results to a CSV file
    output_filename = 'envelope_path.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Move_Number', 'X', 'Y', 'Z']) # Header
        for i, pos in enumerate(recorded_path):
            writer.writerow([i+1] + pos)
            
    print(f"\nPath complete. All {len(recorded_path)} points saved to {output_filename}.")
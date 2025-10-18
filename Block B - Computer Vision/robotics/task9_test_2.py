from sim_class import Simulation
import time

# --- Setup ---
print("Initializing Simulation...")
sim = Simulation(num_agents=1)
print("Simulation Initialized.")

# --- The Challenge: Find all 8 corners ---

# 1. Define all the corner vectors we want to test.
corner_targets = [
    # Action Vector         Corner Name
    ([ 1.0,  1.0,  1.0, 0], "(+X, +Y, +Z)"),
    ([ 1.0,  1.0, -1.0, 0], "(+X, +Y, -Z)"),
    ([ 1.0, -1.0,  1.0, 0], "(+X, -Y, +Z)"),
    ([ 1.0, -1.0, -1.0, 0], "(+X, -Y, -Z)"),
    ([-1.0,  1.0,  1.0, 0], "(-X, +Y, +Z)"),
    ([-1.0,  1.0, -1.0, 0], "(-X, +Y, -Z)"),
    ([-1.0, -1.0,  1.0, 0], "(-X, -Y, +Z)"),
    ([-1.0, -1.0, -1.0, 0], "(-X, -Y, -Z)"),
]

# 2. A dictionary to store the results
working_envelope = {}

# 3. Loop through each target and find its corner
for vector, name in corner_targets:
    print(f"\n--- Finding corner: {name} ---")

    # IMPORTANT: Reset the simulation to start from the center each time
    sim.reset(num_agents=1)

    actions = [vector]
    last_position = None

    while True:
        state = sim.run(actions)
         # --- THE FINAL FIX ---
        # Don't use a hard-coded key. Instead, get the first key from the dictionary.
        robot_key = list(state.keys())[0] 
        # Now use that dynamic key to get the position.
        current_position = state[robot_key]['pipette_position']
        # --- END OF FIX ---

        if current_position == last_position:
            break

        last_position = current_position
        print(f"Current Position: {last_position}", end='\r')

    # Store the final coordinates in our results dictionary
    working_envelope[name] = last_position
    print(f"Corner {name} found at: {last_position}")


# 4. Print the final results
print("\n" + "="*50)
print("              Working Envelope Results")
print("="*50)
for name, position in working_envelope.items():
    print(f"Corner {name}: {position}")
print("="*50)

print("\nScript finished.")
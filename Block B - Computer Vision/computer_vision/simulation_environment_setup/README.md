# Task 9: Robotic Simulation and Workspace Analysis

This project uses the PyBullet physics engine to simulate an Opentrons OT-2 robot. The primary goal is to determine the robot's working envelope by programmatically moving its pipette to the limits of its workspace and recording the coordinates.

---

## Environment Setup & Dependencies

### Dependencies
To run this project, you need a Python environment with the following dependencies installed.

* **Python** (Version 3.10+ recommended)
* **PyBullet**: A real-time physics engine used for the simulation.
* **Git** (For cloning the OT-2 Digital Twin repository)
* **OT-2 Digital Twin Files** (Simulation assets, including `sim_class.py`)


### Installation
You can set up the environment and install the dependencies using the following commands:

```bash
# It is recommended to use a virtual environment (e.g., venv or conda)

# Install PyBullet using pip/ using conda (this is if you do not have Visual Studio installed)
pip install pybullet

conda install -c conda-forge pybullet

# Clone the official repository
# The OT-2 Digital Twin files are required to run the simulation.

git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git

```
## Running the simulation
Once the dependencies are installed and the digital twin files are set up, you can run the simulation:

1. Navigate to the project directory where your simulation script (sim_env.py) is stored:
*cd path/to/your/project*

2. Run the simulation script:
*python sim_env.py*

This script moves the pipette to predefined positions and prints the recorded pipette coordinates.

## Code structure

1. The move_until_stopped Function
This is the core engine of the script. Its only job is to take a single velocity command (e.g., "move left and down") and apply it continuously. It constantly checks the robot's position, and as soon as it detects that the robot is no longer moving, it concludes that it has hit a physical limit and returns the robot's final state.

2. The Main Execution Block
This part orchestrates the entire process:

*Defines the Path*: It starts with a predefined list called movement_sequence. This list acts as a "script" or set of instructions, telling the robot which direction to push itself in for each step of its journey.

*Initializes the Simulation*: It creates the simulation window and the robot.

*Executes the Sequence*: It loops through the movement_sequence, calling the move_until_stopped function for each instruction.

*Records the Data*: After each move, it records the pipette's final coordinates in a list.

*Saves the Results*: Once the entire sequence is complete, it saves the list of all recorded coordinates into a sequential_path.csv file for permanent storage.

In short, the script uses a simple "push-until-stopped" function to execute a pre-programmed sequence of movements, effectively tracing the boundaries of the robot's environment.


```python
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

```

## Working Envelope

The following pipette tip positions were recorded at each step during the simulation:
| Corner   |     X    |     Y    |    Z   |
|----------|----------|----------|--------|
| 1        | -0.1873  | -0.1711  | 0.1195 |
| 2        | 0.253    | -0.1705  | 0.1195 |
| 3        | 0.253    | 0.2199   | 0.1195 |
| 4        | -0.1874  | 0.2195   | 0.1195 |
| 5        | -0.187   | 0.2195   | 0.2902 |
| 6        | -0.187   | -0.1708  | 0.2895 |
| 7        | 0.253    | -0.1705  | 0.2895 |
| 8        | 0.253    |  0.2202  | 0.2895 |
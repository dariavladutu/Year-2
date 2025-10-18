# Custom Reinforcement Learning Environment for OT-2 Robot

This project implements a custom OpenAI Gym environment for controlling an Opentrons OT-2 liquid-handling robot. The goal is to create a standardized interface that allows modern reinforcement learning (RL) algorithms to be trained to perform complex pipetting tasks.

This wrapper, `OT2Env`, communicates with the robot's HTTP API, translating RL actions (e.g., aspirate, dispense) into real-world robot commands and retrieving sensor data (e.g., pipette position and volume) as observations for the RL agent.

## Requirements

This project is built with Python 3.8+ and requires the following libraries.

* **Hardware**: Access to an Opentrons OT-2 robot with its HTTP API server running, or a compatible simulator.
* **Python Libraries**:
    * `gym`: The core OpenAI Gym framework for building environments.
    * `stable-baselines3`: For the reinforcement learning algorithms (PPO, A2C).
    * `numpy`: For numerical operations.
    * `opentrons-http-api-client`: For communication with the OT-2 robot.
    * `torch`: As `stable-baselines3` is built on PyTorch.

## Environment Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    It is recommended to create a `requirements.txt` file with the following content:
    ```txt
    gym
    stable-baselines3[extra]
    numpy
    opentrons-http-api-client
    ```
    Then, install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `test_wrapper.py` script demonstrates how to initialize the custom `OT2Env` and train a Proximal Policy Optimization (PPO) agent from `stable-baselines3` on it.

To run the training demonstration, execute the following command in your terminal:
```bash
python test_wrapper.py
```
This will start the training process, and you will see the progress output from the Stable Baselines3 logger in your console.

## Project Structure

* **`ot2_gym_wrapper.py`**: Contains the core `OT2Env` class. This file defines the environment's action space, observation space, and the `step()` and `reset()` methods that interface with the OT-2 robot.
* **`test_wrapper.py`**: A script that serves as an example of how to use the custom environment. It initializes `OT2Env` and trains a standard RL agent on it.
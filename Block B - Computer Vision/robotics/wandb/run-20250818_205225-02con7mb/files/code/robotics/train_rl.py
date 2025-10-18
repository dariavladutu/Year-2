import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from wandb.integration.sb3 import WandbCallback
import wandb
import os
import argparse # For command-line arguments

# Use the environment wrapper from your friend's code
from ot2_gym_wrapper_2 import OT2Env

def main(args):
    """
    Main function to train a Reinforcement Learning agent for the OT-2 environment.
    This script is configured for a final, high-accuracy tuning run.
    """
    
    # --- Configuration ---
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.total_timesteps,
        "env_name": "OT2Env-v2",
        "algorithm": "PPO",
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
    }

    # --- Weights & Biases Initialization ---
    run = wandb.init(
        project="ot2_rl_tuning_local", 
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    print("--- Starting RL Agent Training (Final Tuning) ---")
    print(f"Timesteps: {config['total_timesteps']}")
    print(f"Hyperparameters: {config}")

    env = OT2Env()

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, # Save less frequently for long runs
        save_path=f"./models/{run.id}",
        name_prefix="rl_model",
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=None,
        verbose=2,
    )

    # --- Learning Rate Schedule ---
    # **MODIFIED**: This will linearly decrease the learning rate from its initial
    # value down to a very small number, encouraging fine-tuning at the end.
    lr_schedule = get_linear_fn(
        start_value=config["learning_rate"],
        end_value=0.00001,
        end_fraction=1.0
    )

    # --- Model Training ---
    model = PPO(
        config["policy_type"],
        env,
        learning_rate=lr_schedule, # Use the schedule instead of a fixed value
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cpu"
    )

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[checkpoint_callback, wandb_callback],
        )
        print("\n--- Training Finished ---")
        final_model_path = f"models/{run.id}/final_model.zip"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
    finally:
        env.close()
        run.finish()

if __name__ == '__main__':
    # --- Argument Parsing ---
    # Using the best hyperparameters found so far.
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="The initial learning rate")
    # Increased timesteps for a final, long run.
    parser.add_argument("--total_timesteps", type=int, default=3000000, help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs when optimizing the surrogate loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance for GAE")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO")
    
    args = parser.parse_args()
    main(args)

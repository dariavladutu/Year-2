import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import os
import argparse

# Use the environment wrapper that supports custom rewards
from ot2_gym_wrapper_2 import OT2Env

def main(args):
    """
    Main function to train a Reinforcement Learning agent.
    This script is fully configurable for a comprehensive hyperparameter search,
    including tuning of the environment's reward function.
    """
    
    # --- Configuration ---
    # All hyperparameters are now passed via args
    config = vars(args)

    # --- Weights & Biases Initialization ---
    run = wandb.init(
        project="ot2_rl_tuning_local", 
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    print("--- Starting RL Agent Hyperparameter Search ---")
    print(f"Timesteps: {config['total_timesteps']}")
    print(f"Hyperparameters: {config}")

    # --- Environment Initialization ---
    # Pass the reward parameters from the arguments directly to the environment
    env = OT2Env(
        threshold=args.threshold,
        reward_distance_scale=args.reward_distance_scale,
        step_penalty=args.step_penalty,
        bonus_reward=args.bonus_reward
    )

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./models/{run.id}",
        name_prefix="rl_model",
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=None,
        verbose=2,
    )

    # --- Model Training ---
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": [args.hidden_units, args.hidden_units]},
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cpu"
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
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
    # Includes both PPO and environment reward hyperparameters
    parser = argparse.ArgumentParser()
    # PPO Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--total_timesteps", type=int, default=3000000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae_lambda", type=float, default=0.92)
    parser.add_argument("--clip_range", type=float, default=0.25)
    parser.add_argument("--hidden_units", type=int, default=128)
    
    # Environment Reward Hyperparameters
    parser.add_argument("--threshold", type=float, default=0.001, help="Success threshold in meters")
    parser.add_argument("--reward_distance_scale", type=int, default=120, help="Multiplier for the distance penalty")
    parser.add_argument("--step_penalty", type=float, default=-0.75, help="Constant penalty applied at each step")
    parser.add_argument("--bonus_reward", type=int, default=90, help="Bonus reward for reaching the target")
    
    args = parser.parse_args()
    main(args)

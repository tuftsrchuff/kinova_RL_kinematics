import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import time as t
import numpy as np
import pybullet as p
import time as time

from env import EmptyScene
from robot import KinovaRobotiq85Sim
from task import GoToTask
from utilities import YCBModels

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def get_task(task_name, args):
    if task_name == "go_to":
        return GoToTask(
            args["robot"].id, args["robot"].eef_id, (0.25, 0.25, 0.25), args["radius"]
        )
    else:
        raise ValueError("Unknown task: {}".format(task_name))

#Run all saved models for 100 episodes and return average rewards
ycb_models = YCBModels(
    os.path.join("./data/ycb", "**", "textured-decmp.obj"),
)
camera = None
robot = KinovaRobotiq85Sim((0, 0, 0), (0, 0, 0))
env = EmptyScene(robot, ycb_models, camera, vis=False)
robot.construct_new_position_actions()
target_task = get_task("go_to", {"robot": robot, "radius": 0.1})
robot.reset_arm_random()

env.set_task(target_task)
env = TimeLimit(env, max_episode_steps=500)
env = Monitor(env, allow_early_resets=True)
check_env(env)
rewards = []

models = 5
save_freq = 5000
model_per_rand = 20
f = open(f"./rewards/rand_target_rewards.txt", "w")
for model_num in range(1,model_per_rand+1):
    mean_reward_all_mod = 0
    for i in range(1, models+1):
        model_string = f"./logs/Rand_target_model_mod_{i}_{save_freq*model_num}_steps.zip"
        print(f"Loading model {model_string}")
        model = PPO.load(model_string, env=env)
        print("Evaluating policy")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)
        print(f"Mean:{mean_reward} Std:{std_reward}")
        mean_reward_all_mod += mean_reward
        rewards.append(mean_reward)
        print(rewards)
        del model

    mean_reward_all_mod = mean_reward_all_mod / models

    f.write(str(mean_reward_all_mod))
    f.write("\n")
    
env.close()
f.close()
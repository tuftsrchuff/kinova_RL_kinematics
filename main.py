import os
import time as t
import numpy as np
import pybullet as p
import time as time

from env import EmptyScene
from robot import KinovaRobotiq85Sim
from task import GoToTask
from utilities import YCBModels
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN, PPO, HER
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback


def run_on_task(model, env, deterministic=True):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(
            obs, deterministic=deterministic
        )  # ignoring states return val
        obs, _, dones, _, _ = env.step(action)  # ignoring reward, info return val
        d = (
            dones if type(dones) is np.ndarray else np.asarray(dones)
        )  # ensure dones is a list
        if d.all():
            print("Done.")
            break


def get_task(task_name, args):
    if task_name == "go_to":
        return GoToTask(
            args["robot"].id, args["robot"].eef_id, (0.25, 0.25, 0.25), args["radius"]
        )
    else:
        raise ValueError("Unknown task: {}".format(task_name))

def learner():
    seeds = [1, 2, 3, 4, 5]
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
    check_env(env)

    model_class = DQN
    goal_selection_strategy = 'future'

    for i in seeds:
        # Initialize the model
        # Wrap the model
        model = HER('MlpPolicy', env, model_class, n_sampled_goal=4,goal_selection_strategy=goal_selection_strategy, verbose=1)

        # model = PPO("MultiInputPolicy", env, verbose=1, n_steps=50, batch_size=100, seed=i)
        # Save a checkpoint every 5000 steps = 20 models per seed
        prefix = f"Rand_arm_model_mod_{i}"
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                            name_prefix=prefix)
        model.set_env(env)

        model.learn(total_timesteps=100000, callback=checkpoint_callback, progress_bar=True)
        del model

    del env
    p.disconnect()
    print("\n\n\nDone")


if __name__ == "__main__":
    learner()

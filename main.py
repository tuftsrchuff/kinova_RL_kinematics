import os

import time as t
import numpy as np
from env import EmptyScene
from robot import KinovaRobotiq85
from task import GoToTask
from utilities import YCBModels, Camera

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def timeout(start_time, seconds):
    return (t.time() - start_time) > seconds


def learn_on_task(model, env, deterministic=False, timeout_seconds=60, step_times=3):
    obs = env.reset()
    start_time = t.time()
    step_time = t.time()
    while True:
        action, _ = model.predict(
            obs, deterministic=deterministic
        )  # ignoring states return val
        obs, rewards, dones, _ = env.step(action)  # ignoring info return val
        d = dones if type(dones) is np.ndarray else np.asarray(dones) # ensure dones is a list
        if d.any() or timeout(step_time, step_times):
            print(np.max(rewards))
            model.train()
            obs = env.reset()
            step_time = t.time()
            print("resume")
        if d.all() or timeout(start_time, timeout_seconds):
            model.train()
            print("Done.")
            break
    return model


def learner():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    robot = KinovaRobotiq85((0, 0, 0), (0, 0, 0))
    env = EmptyScene(robot, ycb_models, camera, vis=False)
    robot.construct_new_position_actions()
    gototask = GoToTask(robot.id, robot.eef_id, (0.25, 0.25, 0.25), 0.25)
    env.set_task(gototask)

    print("make env")
    env = make_vec_env(lambda: env, n_envs=10) # type: ignore

    print("make model")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=10, batch_size=10, seed=0)
    print("setup learn")
    model.learn(total_timesteps=10)

    print("start learn")
    model = learn_on_task(model, env)

    del env
    p.disconnect()
    print("\n\n\nDone")

    env = EmptyScene(robot, ycb_models, camera, vis=True)
    gototask.spawn_target()
    env.set_task(gototask)
    learn_on_task(model, env, step_times=60, deterministic=False)

if __name__ == "__main__":
    learner()

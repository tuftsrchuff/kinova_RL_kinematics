import os

import time as t
import tqdm
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


def run_on_task(model, env, deterministic=True):
    obs = env.reset()
    while True:
        action, _ = model.predict(
            obs, deterministic=deterministic
        )  # ignoring states return val
        obs, _, dones, _ = env.step(action)  # ignoring reward, info return val
        d = (
            dones if type(dones) is np.ndarray else np.asarray(dones)
        )  # ensure dones is a list
        if d.all():
            print("Done.")
            break


def learn_on_task(model, env, deterministic=False, timeout_seconds=60, step_times=3):
    obs = env.reset()
    start_time = t.time()
    step_time = t.time()
    while True:
        action, _ = model.predict(
            obs, deterministic=deterministic
        )  # ignoring states return val
        obs, rewards, dones, _ = env.step(action)  # ignoring info return val
        d = (
            dones if type(dones) is np.ndarray else np.asarray(dones)
        )  # ensure dones is a list
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


def learn_with_timeout(model, timeout_seconds, chunk_size=1000):
    start_time = t.time()
    tqdm_bar = tqdm.tqdm(total=timeout_seconds)
    while not timeout(start_time, timeout_seconds):
        model.learn(total_timesteps=chunk_size, log_interval=None)
        time_progress = int(t.time() - start_time)
        tqdm_bar.update(time_progress)


def learner():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    robot = KinovaRobotiq85((0, 0, 0), (0, 0, 0))
    env = EmptyScene(robot, ycb_models, camera, vis=False)
    robot.construct_new_position_actions()
    gototask = GoToTask(robot.id, robot.eef_id, (0.25, 0.25, 0.25), 0.05)
    env.set_task(gototask)

    print("make env")
    env = make_vec_env(lambda: env, n_envs=10)  # type: ignore
    model = PPO("MlpPolicy", env, verbose=1, n_steps=50, batch_size=100, seed=0)

    print("learn")
    learn_with_timeout(model, 45)

    # print("start learn")
    # model = run_on_task(model, env, deterministic=True)

    del env
    p.disconnect()
    print("\n\n\nDone")

    env = EmptyScene(robot, ycb_models, camera, vis=True)
    gototask.spawn_target()
    env.set_task(gototask)
    run_on_task(model, env, deterministic=False)


if __name__ == "__main__":
    learner()

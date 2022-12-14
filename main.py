import os

import time as t
from env import EmptyScene
from robot import KinovaRobotiq85
from task import GoToTask
from utilities import YCBModels, Camera

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def timeout(start_time, seconds):
    return (t.time() - start_time) > seconds


def learner():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    robot = KinovaRobotiq85((0, 0, 0), (0, 0, 0))
    env = EmptyScene(robot, ycb_models, camera, vis=False)
    robot.extra_action_count = 16
    robot.construct_new_position_actions()
    env.set_task(GoToTask((0.25, 0.25, 0.25), robot.id, robot.eef_id))

    print("make env")
    env = make_vec_env(lambda: env, n_envs=40)

    print("make model")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=20, batch_size=100, seed=0)
    print("setup learn")
    model.learn(total_timesteps=20)
    obs = env.reset()
    start_time = t.time()
    step_time = t.time()

    print("enter loop")
    while True:
        action, _ = model.predict(obs)  # ignoring states return val
        obs, rewards, dones, info = env.step(action)
        if any(dones) or timeout(step_time, 3):
            print(max(rewards))
            model.train()
            obs = env.reset()
            step_time = t.time()
            print('resume')
        if all(dones) or timeout(start_time, 30):
            model.train()
            print("Done.")
            break

    del env, camera
    p.disconnect()
    print(rewards, dones, info)
    print("\n\n\nDone")

    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    env = EmptyScene(robot, ycb_models, camera, vis=True)
    env.set_task(GoToTask((0.25, 0.25, 0.25), robot.id, robot.eef_id))
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=False)  # ignoring states return val
        obs, rewards, dones, info = env.step(action)
        print(rewards)
        if dones:
            obs = env.reset()
            # break


def random_agent():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)

    robot = KinovaRobotiq85((0, 0, 0), (0, 0, 0))
    env = EmptyScene(robot, ycb_models, camera, vis=True)
    env.set_task(GoToTask((0.25, 0.25, 0.25), robot.id, robot.eef_id))

    env = make_vec_env(lambda: env, n_envs=8)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)
    # env.SIMULATION_STEP_DELAY = 0
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)  # ignoring states return val
        obs, rewards, dones, info = env.step(action)
        print(rewards, dones, info)
        env.render()


if __name__ == "__main__":
    learner()

import os

import time as t
import tqdm
import numpy as np
import pybullet as p

from env import EmptyScene
from robot import KinovaRobotiq85
from task import ObjectMoveTask, GoToTask
from utilities import YCBModels, Camera
from violations import CollisionViolation, ActionUndoViolation
from violationcache import ViolationCache
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

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
        obs, _, dones, _ = env.step(action)  # ignoring rewards, info return val
        d = (
            dones if type(dones) is np.ndarray else np.asarray(dones)
        )  # ensure dones is a list
        if d.any() or timeout(step_time, step_times):
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

def learn_with_actioncache(model, env, actioncache, timeout_seconds,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
       callback: MaybeCallback = None,
   ):
    obs = env.reset()
    start_time = t.time()

    _, callback = model._setup_learn(
            5000,#model.total_timesteps,
            model.eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
    )

    bar = tqdm.tqdm(total=timeout_seconds)
    callback.on_training_start(locals(), globals())
    while not timeout(start_time, timeout_seconds):
        bar.update(int(t.time() - start_time))
        action_search = True
        action = None
        while action_search:
            action, _ = model.predict(
                obs, deterministic=False
            )
            if actioncache.in_violation(obs, action):
                action_search = False
        obs, _, dones, _ = env.step(action)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
        model.train()
    callback.on_training_end()

def learner():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    robot = KinovaRobotiq85((0, 0, 0), (0, 0, 0))
    env = EmptyScene(robot, ycb_models, camera, vis=False)
    robot.construct_new_position_actions()
    #target_task = GoToTask(robot.id, robot.eef_id, (0.25, 0.25, 0.25), 0.05)
    #target_task = GoToTask(robot.id, robot.eef_id, (0.25, 0.25, 0.25), 0.25)
    #target_task = CloseGripperTask(robot)
    #target_task = GrabTask(robot, robot.eef_id, (0.25, 0.25, 0.25), 0.25)
    target_task = ObjectMoveTask("./urdf/objects/block.urdf", (0.25, 0, 0.25), (0, 0, 0, 1), robot)
    env.set_task(target_task)

    print("make env")
    env = make_vec_env(lambda: env, n_envs=10)  # type: ignore
    model = PPO("MlpPolicy", env, verbose=1, n_steps=50, batch_size=100, seed=0)

    print("learn")
    learn_with_timeout(model, 5*60)

    del env
    p.disconnect()
    print("\n\n\nDone")

    env = EmptyScene(robot, ycb_models, camera, vis=True)
    target_task.reset()
    env.set_task(target_task)
    run_on_task(model, env, deterministic=False)


if __name__ == "__main__":
    learner()

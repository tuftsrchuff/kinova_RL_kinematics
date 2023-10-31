import os
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
from stable_baselines3.common.callbacks import CheckpointCallback



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




def get_task(task_name, args):
    if task_name == "object_move":
        return ObjectMoveTask(
            "./urdf/objects/block.urdf", (0.25, 0, 0.25), (0, 0, 0, 1), args["robot"]
        )
    elif task_name == "go_to":
        return GoToTask(
            args["robot"].id, args["robot"].eef_id, (0.25, 0.25, 0.25), args["radius"]
        )
    elif task_name == "close_gripper":
        return CloseGripperTask(args["robot"])
    else:
        raise ValueError("Unknown task: {}".format(task_name))


def learner():
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    camera = None
    robot = KinovaRobotiq85Sim((0, 0, 0), (0, 0, 0))
    robot.construct_new_position_actions()
    env = EmptyScene(robot, ycb_models, camera, vis=True)
    env.SIMULATION_STEP_DELAY = 0
    target_task = get_task("go_to", {"robot": robot, "radius": 0.1})
    robot.reset_arm_random()
    env.set_task(target_task)

    model = PPO.load("./logs/Rand_both_model_mod_5_100000_steps.zip")

    obs, _ = env.reset()


    #Use trained model to move arm
    finished = False
    while not finished:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _, info = env.step(action)
        print(action)
        if dones == True:
            print("Done!")
            finished = True
    del env
    p.disconnect()
    print("\n\n\nDone")


if __name__ == "__main__":
    learner()

import time

import numpy as np
import pybullet as p
import pybullet_data
from robot import KinovaRobotiq85

from utilities import Models
from task import Task
from tqdm import tqdm
import gym


class FailToReachTargetError(RuntimeError):
    pass


class EmptyScene(gym.Env):

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(
        self, robot: KinovaRobotiq85, models: Models, camera=None, vis=False
    ) -> None:
        super().__init__()
        self.robot = robot
        self.models = models
        self.vis = vis
        self.camera = camera
        self.steps = 0

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setTimeStep(0.01)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        self.joint_debug_params = []
        self.gripper_opening_length_control = None
        if self.vis:
            self.joint_debug_params = [
                p.addUserDebugParameter(
                    "joint_{}".format(i),
                    -p.getJointInfo(self.robot.id, i)[11],
                    p.getJointInfo(self.robot.id, i)[11],
                    0,
                )
                for i in self.robot.arm_dof_ids
            ]

            self.gripper_opening_length_control = p.addUserDebugParameter(
                "gripper_opening_length", 0, 0.085, 0.04
            )

        self.reward = 0
        self.task = Task()

        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(8,), dtype=np.float32
        )
        # our action space is to select a specific joint and move that joint to the next point.
        # we can move a joint in either direction, and there are 8 joints, so we have 16 possible actions.
        self.base_actions = 16
        self.action_space = gym.spaces.Discrete(self.base_actions + self.robot.extra_action_count)

    def set_task(self, task: Task):
        self.task = task

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        self.steps += 1
        p.stepSimulation()
        #if self.vis:
        #    time.sleep(self.SIMULATION_STEP_DELAY)
        # self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        read_vals = [
            p.readUserDebugParameter(param)
            for param in self.joint_debug_params + [self.gripper_opening_length_control]
        ]

        return read_vals

    def convert_action(self, action):
        # action is one of the 16 possible actions. we can convert this to a joint index and a direction
        # by dividing by 2 and taking the floor and remainder, respectively.
        if action < self.base_actions:
            joint_index = action // 2
            direction = action % 2

            returnme = [0] * 8
            returnme[joint_index] = -1 if direction == 0 else 1
            return returnme
        else:
            return self.robot.convert_action(action)

    def step(self, action):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        if action < self.base_actions:
            joint_index = action // 2
            direction = action % 2

            cvt_action = [0] * 8
            cvt_action[joint_index] = -1 if direction == 0 else 1

            self.robot.move_arm_step(cvt_action)
        else:
            self.robot.move_arm_bonus(action)

        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if self.task.is_done() else False
        return self.get_observation(), reward, done, {}

    def update_reward(self):
        last_reward = self.reward
        self.reward = self.task.reward()
        if last_reward > self.reward:
            return self.reward -1 # punish stepping away from the goal
        return self.reward

    def get_observation(self):
        # construct a unique mapping from joint position to observation
        obs = np.zeros(8)
        for i, _id in enumerate(self.robot.arm_dof_ids):
            raw_value = p.getJointState(self.robot.id, _id)[0]
            obs[i] = raw_value
        return obs

    def reset(self):
        self.steps = 0
        self.reward = 0
        self.robot.reset()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)

    def render(self, mode="human"):
        pass

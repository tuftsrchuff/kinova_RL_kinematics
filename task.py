#!/usr/bin/env python3

import abc
import pybullet as p
import numpy as np


class Task:
    def __init__(self):
        pass

    @abc.abstractmethod
    def reward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_done(self):
        raise NotImplementedError


class GoToTask(Task):
    def __init__(self, robot_id, eef_id, target_pos, target_radius):
        self.target_pos = target_pos
        self.robot_id = robot_id
        self.eef_id = eef_id
        self.target_radius = target_radius
        self.reset()

    def reset(self):
        # spawn a red circle at the target position
        self.target_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.target_radius,
            rgbaColor=[1, 0, 0, 1],
        )
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=self.target_id,
            basePosition=self.target_pos,
        )

    def reward(self):
        ee_pos = p.getLinkState(self.robot_id, self.eef_id)[0]
        # euclidean distance
        return -np.linalg.norm(np.array(ee_pos) - np.array(self.target_pos))

    def is_done(self):
        ee_pos = p.getLinkState(self.robot_id, self.eef_id)[0]
        return (
            np.linalg.norm(np.array(ee_pos) - np.array(self.target_pos))
            < self.target_radius
        )

class CloseGripperTask(Task):
    def __init__(self, robot):
        self.robot = robot

    def reward(self):
        return 1 if self.is_done() else 0

    def is_done(self):
        return self.robot.is_gripper_closed

    def reset(self):
        pass

class ObjectMoveTask(Task):
    def __init__(self, object_path, position, orientation, robot):
        self.object_path = object_path
        self.position = position
        self.orientation = orientation
        self.robot = robot
        self.reset()

    def reward(self):
        object_current_position = p.getBasePositionAndOrientation(self.object_id)[0]
        object_dist_reward = -np.linalg.norm(np.array(object_current_position) - np.array(self.object_start_position))
        gripper_position = p.getLinkState(self.robot.id, self.robot.eef_id)[0]
        gripper_dist_to_object_reward = -np.linalg.norm(np.array(object_current_position) - np.array(gripper_position))
        rounded_dist = round(object_dist_reward + gripper_dist_to_object_reward, 2) # round to 2 decimal places to remove noise
        return rounded_dist

    def is_done(self):
        object_current_position = p.getBasePositionAndOrientation(self.object_id)[0]
        dist = np.linalg.norm(np.array(object_current_position) - np.array(self.object_start_position))
        if dist > 0.25:
            print("Object moved {} meters".format(dist))
            return True
        return False

    def reset(self):
        self.object_id = p.loadURDF(self.object_path, self.position, self.orientation)
        self.object_start_position = p.getBasePositionAndOrientation(self.object_id)[0]

class GrabTask(Task):
    def __init__(self, robot, eef_id, target_pos, target_radius):
        self.index = 0
        self.steps = [GoToTask, CloseGripperTask]
        self.args = [
            (robot.id, eef_id, target_pos, target_radius),
            (robot,),
                ]
        self.current_step = self.steps[self.index](*self.args[self.index])

    def reward(self):
        return self.current_step.reward()

    def is_done(self):
        if self.current_step.is_done():
            if self.index == len(self.steps) - 1:
                print("Done")
                return True
            print("next step")
            self.index += 1
            self.current_step = self.steps[self.index](*self.args[self.index])
            self.reset()
        return False

    def reset(self):
        self.index = 0
        self.current_step = self.steps[self.index](*self.args[self.index])
        self.current_step.reset()

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

    def get_target_pos(self):
        return self.target_pos
    
    def set_target_pos(self, target_pos)->None:
        self.target_pos = target_pos

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

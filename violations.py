#!/usr/bin/env python3

import abc
import pybullet as p
from typing import Dict, List

class Violation(metaclass=abc.ABCMeta):
    def __init__(self, args:Dict):
        self.args = args

    @abc.abstractmethod
    def in_violation(self, observation, propsed_action) -> bool:
        raise NotImplementedError

class CollisionViolation(Violation):
    def __init__(self, args:Dict):
        required_args = ['joint_ids', 'object_id_self', 'object_ids_other']
        assert "joint_ids" in args, "Missing required arg from: {}".format(required_args)
        assert "object_id_self" in args, "Missing required arg from: {}".format(required_args)
        assert "object_ids_env" in args, "Missing required arg from: {}".format(required_args)
        assert type(args["joint_ids"]) == list, "Invalid argument type: joint_ids must be a list"
        assert type(args["object_ids_env"]) == list, "Invalid argument type: object_ids_env must be a list"
        assert type(args["object_id_self"]) == int, "Invalid argument type: object_id_self must be a float"
        self.self_id = args["object_id_self"]
        self.object_ids_env = args["object_ids_env"]
        self.joint_ids = args["joint_ids"]

    def in_violation(self, _, proposed_action) -> bool:
        return len(self._find_collisions(proposed_action)) > 0

    def find_active_collision(self):
        collisions = []
        for obj_id in self.object_ids_env:
            if p.getContactPoints(self.self_id, obj_id):
                collisions.append(obj_id)
        return collisions

    def _find_collisions(self, joint_states:List):
        # save current joint state configuration
        current_joint_states = p.getJointStates(self.self_id, range(p.getNumJoints(self.self_id)))
        # now set the joint states to the proposed joint states
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.self_id, joint_id, joint_states[i])
        # now check for collisions
        collisions = []
        for obj_id in self.object_ids_env:
            if p.getContactPoints(self.self_id, obj_id):
                collisions.append(obj_id)
        # reset the joint states to the original joint states
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.self_id, joint_id, current_joint_states[i][0])
        return collisions

class ActionUndoViolation(Violation):
    def __init__(self, args:Dict):
        required_args = ['robot']
        assert "robot" in args, "Missing required arg from: {}".format(required_args)
        self.robot = args["robot"]

    def in_violation(self, _, proposed_action) -> bool:
        a = self.robot.last_action
        b = proposed_action
        return sum([a[i]+b[i] for i in range(len(a))]) == 0

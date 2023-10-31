from abc import ABC, abstractmethod
from collections import namedtuple
import math
from typing import List
import numpy as np
import pybullet as p
import random

# MOVE_CHUNK = (np.pi / 10)
MOVE_CHUNK_COUNT = 60


class KinovaRobotiq85(ABC):
    def __init__(self, pos, ori, extra_action_count=0, start_joint_angles=None):
        self.base_pos = pos
        self.base_ori_rpy = ori
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.id = None
        self.eef_id = None
        self.extra_action_count = extra_action_count
        self.bonus_actions = []
        self.is_gripper_closed = False
        self.start_joint_angles = start_joint_angles

        self.arm_num_dofs = 7
        self.arm_lower_limits = []
        self.arm_upper_limits = []

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def step_simulation(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        pass

    def reset_gripper(self):
        self.open_gripper()

    @abstractmethod
    def open_gripper(self):
        pass

    @abstractmethod
    def close_gripper(self):
        pass

    @abstractmethod
    def in_collision(self):
        pass

    @abstractmethod
    def violates_limits(self, target_joint_positions):
        pass

    def construct_new_position_actions(self):
        self.bonus_actions = []
        for _ in range(self.extra_action_count):
            action = np.zeros(self.arm_num_dofs)
            for n in range(self.arm_num_dofs):
                joint_pos = np.random.uniform(
                    self.arm_lower_limits[n], self.arm_upper_limits[n]
                )
                action[n] = joint_pos
            self.bonus_actions.append(action)

    @abstractmethod
    def move_gripper(self, open_length):
        pass

    @abstractmethod
    def get_joint_states(self) -> List[float]:
        pass

    @abstractmethod
    def goto_joint_states(self, target_joint_positions: List[float]):
        pass

    def move_arm_step(self, action):
        self.last_action = action
        current_joint_positions = self.get_joint_states()
        target_joint_positions = [
            current_joint_positions[i] + (action[i] * 1 / MOVE_CHUNK_COUNT)
            for i in range(len(current_joint_positions))
        ]
        return self.goto_joint_states(target_joint_positions)


class KinovaRobotiq85Sim(KinovaRobotiq85):
    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()

    def step_simulation(self):
        raise RuntimeError(
            "`step_simulation` method of RobotBase Class should be hooked by the environment."
        )

    def __parse_joint_info__(self):
        assert self.id is not None
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[
                2
            ]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(
                    self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
            info = jointInfo(
                jointID,
                jointName,
                jointType,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                controllable,
            )
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]

        self.arm_lower_limits = [
            info.lowerLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_upper_limits = [
            info.upperLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            info.upperLimit - info.lowerLimit
            for info in self.joints
            if info.controllable
        ][: self.arm_num_dofs]

    def reset_arm_random(self) -> None:
        for lower, upper, joint_id in zip(
            self.arm_lower_limits, self.arm_upper_limits, self.arm_controllable_joints
        ):
            rand_joint_pose = random.uniform(lower, upper)
            p.resetJointState(self.id, joint_id, rand_joint_pose)

            # Wait for a few steps
            for _ in range(10):
                self.step_simulation()


    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(
            self.arm_rest_poses, self.arm_controllable_joints
        ):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])
        self.is_gripper_closed = False

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])
        self.is_gripper_closed = True

    def __init_robot__(self):
        self.arm_num_dofs = 7
        self.eef_id = 8
        self.arm_rest_poses = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        self.arm_dof_ids = [1, 2, 3, 4, 5, 6, 7]

        self.id = p.loadURDF(
            "./urdf/kinova_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.gripper_range = [0, 0.085]

    def get_joint_states(self):
        joint_states = p.getJointStates(self.id, self.arm_controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        return joint_positions

    def goto_joint_states(self, joint_states):
        for joint_state, joint_id in zip(joint_states, self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=joint_state,
                force=self.joints[joint_id].maxForce * 100,
                maxVelocity=self.joints[joint_id].maxVelocity,
            )
        
        return True

    def in_collision(self):
        return p.getContactPoints(self.id) != []

    def violates_limits(self, target_joint_positions):
        return any(
            [
                target_joint_positions[i] < self.arm_lower_limits[i]
                or target_joint_positions[i] > self.arm_upper_limits[i]
                for i in range(self.arm_num_dofs)
            ]
        )

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin(
            (open_length - 0.010) / 0.1143
        )  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity,
        )
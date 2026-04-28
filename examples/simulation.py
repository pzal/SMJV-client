import time

import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.base import MujocoEnv

from smjv_client.quest_publisher import QuestPublisher


QUEST_IP = "10.1.10.100"
FPS = 20

GRIP_THRESHOLD = 0.5
TRIGGER_THRESHOLD = 0.5
Kp_pos = 20.0
Kp_rot = 5.0


def hide_arena_floor_and_walls(env: MujocoEnv) -> None:
    for name in [
        "floor",
        "wall_leftcorner_visual",
        "wall_rightcorner_visual",
        "wall_left_visual",
        "wall_right_visual",
        "wall_rear_visual",
        "wall_front_visual",
    ]:
        try:
            gid = env.sim.model.geom_name2id(name)
            env.sim.model.geom_group[gid] = 2
        except ValueError:
            pass


class WallTimeSyncer:
    def __init__(self, hz: float):
        self._step_period = 1.0 / hz
        self._next_deadline = time.monotonic() + self._step_period

    def sync(self):
        sleep_for = self._next_deadline - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._next_deadline += self._step_period


if __name__ == "__main__":
    env = Lift(
        robots="Panda",
        ignore_done=True,
        control_freq=FPS,
    )
    obs = env.reset()

    hide_arena_floor_and_walls(env)
    quest_publisher = QuestPublisher(env, quest_ip=QUEST_IP)

    a_pressed = False
    ctrl_anchor_pos = None
    ctrl_anchor_rot = None
    robot_anchor_pos = None
    robot_anchor_rot = None
    gripper = -1.0

    syncer = WallTimeSyncer(FPS)

    try:
        while True:
            prev_a = a_pressed

            syncer.sync()
            quest_publisher.publish_state()
            data = quest_publisher.consume_latest_input()

            a_pressed = bool(data["A"])

            right = data["right"]

            grip_held = right["hand_trigger"] > GRIP_THRESHOLD
            trigger_held = right["index_trigger"] > TRIGGER_THRESHOLD

            # A: reset env (rising edge)
            if a_pressed and not prev_a:
                obs = env.reset()
                hide_arena_floor_and_walls(env)
                quest_publisher.refresh(env)

                ctrl_anchor_pos = None
                ctrl_anchor_rot = None
                robot_anchor_pos = None
                robot_anchor_rot = None
                gripper = -1.0
                continue

            # Not holding grip: idle, no env.step
            if not grip_held:
                ctrl_anchor_pos = None
                ctrl_anchor_rot = None
                robot_anchor_pos = None
                robot_anchor_rot = None
                continue

            # First frame of grip: capture anchors
            if ctrl_anchor_pos is None:
                ctrl_anchor_pos = np.asarray(right["pos"], dtype=float).copy()
                ctrl_anchor_rot = Rotation.from_quat(right["rot"])
                robot_anchor_pos = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
                robot_anchor_rot = Rotation.from_quat(obs["robot0_eef_quat_site"])
                continue

            ctrl_pos = np.asarray(right["pos"], dtype=float)
            ctrl_rot = Rotation.from_quat(right["rot"])

            target_pos = robot_anchor_pos + (ctrl_pos - ctrl_anchor_pos)
            target_rot = (ctrl_rot * ctrl_anchor_rot.inv()) * robot_anchor_rot

            robot_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
            robot_rot = Rotation.from_quat(obs["robot0_eef_quat_site"])
            action_pos = Kp_pos * (target_pos - robot_pos)
            action_rot = Kp_rot * (target_rot * robot_rot.inv()).as_rotvec()

            gripper = 1.0 if trigger_held else -1.0
            action = np.concatenate([action_pos, action_rot, [gripper]])

            obs, _, _, _ = env.step(action)

    except KeyboardInterrupt:
        pass
    finally:
        quest_publisher.close()

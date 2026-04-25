from smjv_client.vr import hide_arena_floor_and_walls
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.lift import Lift

from smjv_client.quest_publisher import QuestPublisher


class WallTimeSyncer:
    def __init__(self, hz: float):
        self._step_period = 1.0 / hz
        self._next_deadline = time.monotonic() + self._step_period

    def sync(self):
        sleep_for = self._next_deadline - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._next_deadline += self._step_period


# Example usage
if __name__ == "__main__":
    FPS = 20

    env = Lift(
        robots="Panda",
        ignore_done=True,
        control_freq=FPS,
    )
    obs = env.reset()
    hide_arena_floor_and_walls(env)
    publisher = QuestPublisher(env)

    current_pos = obs["robot0_eef_pos"]
    delta = np.array([0, 0.35, 0])
    static_delta = np.array([0.2, 0, 0])
    target_pos_1 = current_pos + delta + static_delta
    target_pos_2 = current_pos - delta + static_delta
    target_pos = target_pos_1
    _rot_axis = np.array([1, 1, 0])
    target_ori = _rot_axis / np.linalg.norm(_rot_axis) * np.pi

    step = 0
    syncer = WallTimeSyncer(FPS)
    t0 = time.time()

    pbar = tqdm(desc="steps")
    while True:
        syncer.sync()
        pbar.update(1)
        pbar.display()

        if time.time() > t0 + 5:
            t0 = time.time()
            target_pos = target_pos_1 if target_pos is target_pos_2 else target_pos_2

        current_pos = obs["robot0_eef_pos"]
        current_ori = R.from_quat(obs["robot0_eef_quat_site"])
        action_pos = target_pos - current_pos
        action_ori = (R.from_rotvec(target_ori) * current_ori.inv()).as_rotvec()
        obs, _, _, _ = env.step(np.concat([action_pos, action_ori, [0]]))
        publisher.publish_state()

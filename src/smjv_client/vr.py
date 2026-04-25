from robosuite.environments.base import MujocoEnv



def hide_arena_floor_and_walls(env: MujocoEnv) -> None:
    """Move floor/wall geoms to group 2 so the VR publisher (group 1 only) skips them."""
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

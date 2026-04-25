import asyncio

import msgpack
import websockets
from mujoco import mj_name2id, mjtObj
from simpub.parser.mj import MjModelParser

QUEST_IP = "127.0.0.1"  # fill in your Quest's IP, e.g. "192.168.1.42"
PORT = 8765


class QuestPublisher:
    """Persistent WebSocket; sends scene once on connect, poses each step."""

    def __init__(self, env, visible_geoms_groups=range(5)):
        self.model = env.sim.model._model
        self.data = env.sim.data._data

        sim_scene = MjModelParser(self.model, visible_geoms_groups=[1]).parse()

        flat = []

        def walk(node):
            if node.data is not None:
                flat.append(node.data)
            for c in node.children:
                walk(c)

        walk(sim_scene.root)

        self.tracked = {}
        for so in flat:
            bid = mj_name2id(self.model, mjtObj.mjOBJ_BODY, so["name"])
            if bid >= 0:
                self.tracked[so["name"]] = (self.data.xpos[bid], self.data.xquat[bid])

        self._loop = asyncio.new_event_loop()
        self._ws = self._loop.run_until_complete(self._connect())
        scene_payload = msgpack.packb(
            {"config": sim_scene.config, "objects": flat}, use_bin_type=True
        )
        self._loop.run_until_complete(self._send("scene", scene_payload))

    async def _connect(self):
        return await websockets.connect(
            f"ws://{QUEST_IP}:{PORT}/sim", max_size=64 * 1024 * 1024
        )

    async def _send(self, msg_type: str, data: bytes):
        envelope = msgpack.packb({"type": msg_type, "data": data}, use_bin_type=True)
        await self._ws.send(envelope)

    def publish_state(self):
        state = {}
        for name, (p, q) in self.tracked.items():
            state[name] = [
                float(-p[1]),
                float(p[2]),
                float(p[0]),
                float(q[2]),
                float(-q[3]),
                float(-q[1]),
                float(q[0]),
            ]
        payload = msgpack.packb({"data": state}, use_bin_type=True)
        self._loop.run_until_complete(self._send("poses", payload))

    def close(self):
        async def _close():
            try:
                await self._send("clear", b"")
            except Exception:
                pass
            await self._ws.close()

        self._loop.run_until_complete(_close())
        self._loop.close()

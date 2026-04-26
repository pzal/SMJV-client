import asyncio
import threading
from copy import deepcopy

import msgpack
import websockets
from mujoco import mj_name2id, mjtObj
from simpub.parser.mj import MjModelParser

QUEST_IP = "10.1.20.101"  # fill in your Quest's IP, e.g. "192.168.1.42"
QUEST_IP = "10.1.10.100"  # fill in your Quest's IP, e.g. "192.168.1.42"
PORT = 8765

_LATCHED_BUTTONS = ("A", "B", "X", "Y")
_LATCHED_PER_HAND = ("thumbstick_click",)


class QuestPublisher:
    """Persistent WebSocket; sends scene + per-step poses, receives controller input.

    The asyncio loop runs on a dedicated thread so outbound publishes and the
    inbound recv task share the same socket without blocking each other.
    """

    def __init__(self, env, quest_ip=QUEST_IP, quest_port=PORT, visible_geoms_groups=range(5)):
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.quest_ip = quest_ip
        self.quest_port = quest_port

        sim_scene = MjModelParser(self.model, visible_geoms_groups=[1]).parse()
        self._sim_scene = sim_scene

        flat = []

        def walk(node):
            if node.data is not None:
                flat.append(node.data)
            for c in node.children:
                walk(c)

        walk(sim_scene.root)
        self._flat = flat

        self.tracked = {}
        for so in flat:
            bid = mj_name2id(self.model, mjtObj.mjOBJ_BODY, so["name"])
            if bid >= 0:
                self.tracked[so["name"]] = (self.data.xpos[bid], self.data.xquat[bid])

        self._latest_input = None
        self._input_lock = threading.Lock()

        # Asyncio loop on a background thread; outbound publishes and inbound
        # recv share the same socket without blocking each other.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._loop_thread.start()

        self._ws = self._run(self._connect())
        scene_payload = msgpack.packb(
            {"config": sim_scene.config, "objects": flat}, use_bin_type=True
        )
        self._run(self._send("scene", scene_payload))

        # Long-running coroutine: schedule and keep the Future for later cancel,
        # but DO NOT call .result() — that would block this thread forever.
        self._recv_task = asyncio.run_coroutine_threadsafe(
            self._recv_loop(), self._loop
        )

    def _run(self, coro):
        """Run a short coroutine on the loop thread and block until it returns."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _connect(self):
        return await websockets.connect(
            f"ws://{self.quest_ip}:{self.quest_port}/sim", max_size=64 * 1024 * 1024
        )

    async def _send(self, msg_type: str, data: bytes):
        envelope = msgpack.packb({"type": msg_type, "data": data}, use_bin_type=True)
        await self._ws.send(envelope)

    async def _recv_loop(self):
        try:
            async for msg in self._ws:
                try:
                    envelope = msgpack.unpackb(msg, raw=False)
                except Exception:
                    continue
                if envelope.get("type") != "input":
                    continue
                payload = msgpack.unpackb(envelope["data"], raw=False)
                self._apply_input(payload)
        except websockets.ConnectionClosed:
            pass

    def _apply_input(self, payload):
        with self._input_lock:
            if self._latest_input is None:
                # First frame: take it as-is so analog values seed correctly.
                self._latest_input = payload
                return
            cur = self._latest_input
            for k in _LATCHED_BUTTONS:
                cur[k] = bool(cur.get(k, False)) or bool(payload.get(k, False))
            for hand in ("left", "right"):
                cur_hand = cur.setdefault(hand, {})
                new_hand = payload.get(hand, {})
                for k, v in new_hand.items():
                    if k in _LATCHED_PER_HAND:
                        cur_hand[k] = bool(cur_hand.get(k, False)) or bool(v)
                    else:
                        cur_hand[k] = v

    def consume_latest_input(self):
        """Return a snapshot of the latest input and clear all latched booleans.

        Returns None if no input frame has arrived yet. Analog values (poses,
        triggers, thumbsticks) reflect the latest sample; booleans (A/B/X/Y,
        thumbstick_click) are True if pressed in any frame since the last call.
        """
        with self._input_lock:
            if self._latest_input is None:
                return None
            snapshot = deepcopy(self._latest_input)
            for k in _LATCHED_BUTTONS:
                self._latest_input[k] = False
            for hand in ("left", "right"):
                hand_dict = self._latest_input.get(hand)
                if hand_dict is None:
                    continue
                for k in _LATCHED_PER_HAND:
                    if k in hand_dict:
                        hand_dict[k] = False
            return snapshot

    def send_display(self, value, label: str = ""):
        payload = msgpack.packb(
            {"label": label, "value": str(value)}, use_bin_type=True
        )
        self._run(self._send("display", payload))

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
        self._run(self._send("poses", payload))

    def refresh(self, env) -> None:
        """Re-bind to env's fresh mj_model/mj_data after a hard reset and resend the scene.

        robosuite's default ``hard_reset=True`` on ``env.reset()`` replaces
        ``env.sim`` (and thus the raw mj_model/mj_data) on every reset,
        orphaning the numpy views grabbed at construction time.  Call this
        right after ``env.reset()`` to restore a working state.
        """
        self.model = env.sim.model._model
        self.data = env.sim.data._data

        self.tracked = {}
        for so in self._flat:
            bid = mj_name2id(self.model, mjtObj.mjOBJ_BODY, so["name"])
            if bid >= 0:
                self.tracked[so["name"]] = (self.data.xpos[bid], self.data.xquat[bid])

        scene_payload = msgpack.packb(
            {"config": self._sim_scene.config, "objects": self._flat}, use_bin_type=True
        )
        self._run(self._send("scene", scene_payload))

    def close(self):
        async def _close():
            try:
                await self._send("clear", b"")
            except Exception:
                pass
            await self._ws.close()

        try:
            self._recv_task.cancel()
        except Exception:
            pass
        try:
            self._run(_close())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2)

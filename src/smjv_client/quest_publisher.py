import asyncio
import logging
import threading
from copy import deepcopy

import msgpack
import websockets
from mujoco import mj_id2name, mj_name2id, mjtObj
from simpub.parser.mj import MjModelParser

logger = logging.getLogger(__name__)

PORT = 8765

_LATCHED_PER_HAND = ("a", "b", "x", "y", "thumbstick_click")


def _connection_closed_details(exc: websockets.ConnectionClosed) -> tuple:
    close = exc.rcvd or exc.sent
    code = getattr(close, "code", None)
    reason = getattr(close, "reason", "")
    return code, reason, type(exc).__name__, str(exc)


class QuestPublisher:
    """Persistent WebSocket; sends scene + per-step poses, receives controller input.

    The asyncio loop runs on a dedicated thread so outbound publishes and the
    inbound recv task share the same socket without blocking each other.
    """

    def __init__(self, env, quest_ip, quest_port=PORT, visible_geoms_groups=range(5)):
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.quest_ip = quest_ip
        self.quest_port = quest_port

        self._parse_scene()
        self._rebuild_tracked()
        self._structure_fp = self._structure_fingerprint(self.model)

        self._latest_input = None
        self._input_lock = threading.Lock()
        # Latched booleans use False->True edge detection so a held button
        # produces only one rising edge per physical press, even if the host
        # consumes input multiple times while the button is still down.
        self._pending: dict[str, set[str]] = {"left": set(), "right": set()}

        # Asyncio loop on a background thread; outbound publishes and inbound
        # recv share the same socket without blocking each other.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self._ws = self._run(self._connect())
        self._run(self._send("scene", self._scene_payload()))

        # Long-running coroutine: schedule and keep the Future for later cancel,
        # but DO NOT call .result() — that would block this thread forever.
        self._recv_task = asyncio.run_coroutine_threadsafe(
            self._recv_loop(), self._loop
        )

    def _run(self, coro):
        """Run a short coroutine on the loop thread and block until it returns."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _parse_scene(self) -> None:
        sim_scene = MjModelParser(self.model, visible_geoms_groups=[1]).parse()
        # MjModelParser emits a node for every body, including unnamed ones
        # (e.g. the anonymous wrapper body that robosuite XMLObjects nest their
        # named "object" body inside). Such nodes are unaddressable downstream
        # — no key for per-frame pose updates, no name for Unity to reference —
        # so we splice them out: lift their visuals into the parent, reparent
        # their children to the grandparent, and drop the node.
        self._splice_anonymous_bodies(sim_scene.root)

        flat = []

        def walk(node):
            if node.data is not None:
                flat.append(node.data)
            for c in node.children:
                walk(c)

        walk(sim_scene.root)
        self._sim_scene = sim_scene
        self._flat = flat

    @staticmethod
    def _compose_transforms(outer, inner):
        """Return outer * inner (Unity convention: pos=[x,y,z], rot=[x,y,z,w])."""
        ox, oy, oz = outer["pos"]
        qx, qy, qz, qw = outer["rot"]
        ix, iy, iz = inner["pos"]
        # Rotate inner position by outer quaternion.
        tx = 2.0 * (qy * iz - qz * iy)
        ty = 2.0 * (qz * ix - qx * iz)
        tz = 2.0 * (qx * iy - qy * ix)
        rx = ix + qw * tx + (qy * tz - qz * ty)
        ry = iy + qw * ty + (qz * tx - qx * tz)
        rz = iz + qw * tz + (qx * ty - qy * tx)
        new_pos = [ox + rx, oy + ry, oz + rz]
        # Hamilton quaternion product outer * inner.
        ax, ay, az, aw = inner["rot"]
        new_rot = [
            qw * ax + qx * aw + qy * az - qz * ay,
            qw * ay - qx * az + qy * aw + qz * ax,
            qw * az + qx * ay - qy * ax + qz * aw,
            qw * aw - qx * ax - qy * ay - qz * az,
        ]
        new_scale = [outer["scale"][i] * inner["scale"][i] for i in range(3)]
        return {"pos": new_pos, "rot": new_rot, "scale": new_scale}

    def _splice_anonymous_bodies(self, node) -> None:
        """Remove unnamed bodies from the tree, lifting their content into the parent.

        Visuals and children of an anonymous body get their local transforms
        composed with the body's own transform, then are reattached to the
        grandparent. Post-order so nested anonymous bodies collapse cleanly.
        """
        for child in node.children:
            self._splice_anonymous_bodies(child)

        new_children = []
        for child in node.children:
            if child.data is None or child.data.get("name") is not None:
                new_children.append(child)
                continue

            anon_trans = child.data["trans"]
            if node.data is not None:
                for visual in child.data.get("visuals", []):
                    visual["trans"] = self._compose_transforms(
                        anon_trans, visual["trans"]
                    )
                    node.data["visuals"].append(visual)

            new_parent_name = node.data["name"] if node.data is not None else "root"
            for grandchild in child.children:
                if grandchild.data is not None:
                    grandchild.data["trans"] = self._compose_transforms(
                        anon_trans, grandchild.data["trans"]
                    )
                    grandchild.data["parent"] = new_parent_name
                new_children.append(grandchild)

        node.children = new_children

    def _rebuild_tracked(self) -> None:
        self.tracked = {}
        for so in self._flat:
            bid = mj_name2id(self.model, mjtObj.mjOBJ_BODY, so["name"])
            if bid >= 0:
                self.tracked[so["name"]] = (self.data.xpos[bid], self.data.xquat[bid])

    def _scene_payload(self) -> bytes:
        return msgpack.packb(
            {"config": self._sim_scene.config, "objects": self._flat}, use_bin_type=True
        )

    async def _connect(self):
        return await websockets.connect(
            f"ws://{self.quest_ip}:{self.quest_port}/sim", max_size=64 * 1024 * 1024
        )

    async def _send(self, msg_type: str, data: bytes):
        envelope = msgpack.packb({"type": msg_type, "data": data}, use_bin_type=True)
        try:
            await self._ws.send(envelope)
        except websockets.ConnectionClosed as exc:
            code, reason, exc_type, detail = _connection_closed_details(exc)
            logger.warning(
                "Quest websocket send failed for msg_type=%r: code=%s reason=%r type=%s detail=%s",
                msg_type,
                code,
                reason,
                exc_type,
                detail,
            )
            raise
        except Exception:
            logger.exception(
                "Quest websocket send failed for msg_type=%r",
                msg_type,
            )
            raise

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
        except websockets.ConnectionClosed as exc:
            code, reason, exc_type, detail = _connection_closed_details(exc)
            logger.warning(
                "Quest websocket receive loop closed: code=%s reason=%r type=%s detail=%s",
                code,
                reason,
                exc_type,
                detail,
            )

    def _apply_input(self, payload):
        with self._input_lock:
            # Detect False->True edges for latched buttons against the last raw
            # value seen on the wire, so a held button doesn't re-trigger after
            # consume_latest_input().
            prev = self._latest_input or {}
            for hand in ("left", "right"):
                new_hand = payload.get(hand, {})
                prev_hand = prev.get(hand, {})
                for k in _LATCHED_PER_HAND:
                    if new_hand.get(k) and not prev_hand.get(k):
                        self._pending[hand].add(k)

            self._latest_input = payload

    def consume_latest_input(self):
        """Return a snapshot of the latest input and clear all latched booleans.

        Returns None if no input frame has arrived yet. Analog values (poses,
        triggers, thumbsticks) reflect the latest sample; per-hand latched
        booleans (a, b on right; x, y on left; thumbstick_click on both) are
        True if pressed in any frame since the last call.

        ``rot`` on each hand is returned as ``[x, y, z, w]`` (scipy convention)
        — the Unity app sends ``[w, x, y, z]`` which is reordered here so
        callers can pass it directly to ``Rotation.from_quat()``.
        """
        with self._input_lock:
            if self._latest_input is None:
                return None
            snapshot = deepcopy(self._latest_input)
            # Latched booleans reflect rising edges since the last consume,
            # not the current raw wire value, so a held button only fires once.
            for hand in ("left", "right"):
                hand_snap = snapshot.get(hand)
                if hand_snap is None:
                    continue
                pending = self._pending[hand]
                for k in _LATCHED_PER_HAND:
                    if k in hand_snap:
                        hand_snap[k] = k in pending
                pending.clear()
            # Unity wire format: rot = [w, x, y, z].  Reorder to scipy [x, y, z, w].
            for hand in ("left", "right"):
                hand_snap = snapshot.get(hand)
                if hand_snap and "rot" in hand_snap:
                    w, x, y, z = hand_snap["rot"]
                    hand_snap["rot"] = [x, y, z, w]
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

    @staticmethod
    def _structure_fingerprint(model) -> tuple:
        body_names = tuple(
            mj_id2name(model, mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
        )
        return (model.nbody, model.ngeom, model.nmesh, model.nq, body_names)

    def rebind(self, env) -> None:
        """Re-bind to env's fresh mj_model/mj_data after a hard reset, no network I/O.

        robosuite's default ``hard_reset=True`` on ``env.reset()`` replaces
        ``env.sim`` (and thus the raw mj_model/mj_data) on every reset,
        orphaning the numpy views grabbed at construction time.  Use this
        when scene structure is known to be unchanged — the next
        ``publish_state()`` will paint the new poses.
        """
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self._rebuild_tracked()

    def refresh(self, env, force: bool = False) -> None:
        """Resync after ``env.reset()``.

        With ``force=False`` (default), if the new env's scene structure
        matches the one captured at construction, only re-bind numpy views
        — no scene resend, no Unity-side rebuild, no visible flash.  If the
        structure changed, fall back to a full rebind + scene resend with a
        warning.

        With ``force=True``, always do the full rebind + scene resend.
        Use as an explicit escape hatch.
        """
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        new_fp = self._structure_fingerprint(self.model)

        if not force:
            if new_fp == self._structure_fp:
                self._rebuild_tracked()
                logger.info(
                    "Scene structure unchanged; rebinding only, reusing existing Quest scene."
                )
                return
            logger.warning(
                "Scene structure changed across reset; falling back to full scene resend."
            )
        else:
            logger.info("Force-refreshing Quest scene.")

        self._parse_scene()
        self._rebuild_tracked()
        self._structure_fp = new_fp
        self._run(self._send("scene", self._scene_payload()))

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

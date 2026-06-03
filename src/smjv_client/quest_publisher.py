import asyncio
import hashlib
import logging
import threading
from copy import deepcopy

import msgpack
import websockets
from mujoco import mj_name2id, mjtObj

from .mj_parser import MjSceneParser

logger = logging.getLogger(__name__)

PORT = 8765

_LATCHED_PER_HAND = ("a", "b", "x", "y", "thumbstick_click")

# Reconnect backoff bounds (seconds).
_INITIAL_BACKOFF = 0.5
_MAX_BACKOFF = 10.0

# websockets defaults can take tens of seconds to detect a dead link. Keep
# this short enough for VR recording, but not so short that brief Wi-Fi jitter
# causes avoidable reconnects.
_PING_INTERVAL = 2.0
_PING_TIMEOUT = 2.0
_CLOSE_TIMEOUT = 1.0

_ASSET_KINDS = ("meshes", "textures", "materials")


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

    def __init__(self, env, quest_ip, quest_port=PORT, visible_geoms_groups=(1,)):
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.quest_ip = quest_ip
        self.quest_port = quest_port
        self.visible_geoms_groups = tuple(int(group) for group in visible_geoms_groups)

        self._parse_scene()
        self._rebuild_tracked()
        self._asset_store = self._empty_asset_store()
        self._scene_hash = None

        self._latest_input = None
        self._input_lock = threading.Lock()
        # Latched booleans use False->True edge detection so a held button
        # produces only one rising edge per physical press, even if the host
        # consumes input multiple times while the button is still down.
        self._pending: dict[str, set[str]] = {"left": set(), "right": set()}

        # Set while a live socket is usable; cleared the moment a drop is
        # detected (by either the send or recv side). threading.Event so the
        # caller thread can poll is_connected() cheaply, without touching the
        # loop thread.
        self._connected = threading.Event()
        self._closing = False

        # Asyncio loop on a background thread; outbound publishes and inbound
        # recv share the same socket without blocking each other.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Initial connect stays synchronous so construction fails loudly if the
        # Quest is unreachable at startup.
        try:
            self._ws = self._run(self._connect())
            self._run(self._send_scene_manifest_raw())
        except Exception:
            self._closing = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2)
            raise
        self._connected.set()

        # Long-running supervisor: runs the recv loop and, when the socket
        # drops, reconnects (forever, with backoff) and resends the scene.
        # Schedule and keep the Future for later cancel, but DO NOT call
        # .result() — that would block this thread forever.
        self._supervisor_task = asyncio.run_coroutine_threadsafe(
            self._supervise(), self._loop
        )

    def _run(self, coro):
        """Run a short coroutine on the loop thread and block until it returns."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _parse_scene(self) -> None:
        sim_scene = MjSceneParser(
            self.model, self.visible_geoms_groups
        ).parse()
        # MjSceneParser emits a node for every body, including unnamed ones
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
        # Mesh orientation (mj2unity remap) is baked into MjSceneParser, so no
        # post-hoc realignment is needed here.

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

    @staticmethod
    def _empty_asset_store() -> dict[str, dict[str, dict]]:
        return {kind: {} for kind in _ASSET_KINDS}

    @staticmethod
    def _stable_hash(value) -> str:
        packed = msgpack.packb(value, use_bin_type=True)
        return hashlib.sha256(packed).hexdigest()

    @classmethod
    def _build_scene_manifest_document(
        cls,
        config,
        objects,
    ) -> tuple[dict, dict[str, dict[str, dict]]]:
        """Return a hash-only scene manifest plus a local asset lookup table."""
        assets = cls._empty_asset_store()
        manifest_objects = []

        for obj in objects:
            manifest_obj = deepcopy(obj)
            cls._replace_visual_assets_with_refs(
                manifest_obj,
                assets,
            )
            manifest_obj["contentHash"] = cls._object_content_hash(manifest_obj)
            manifest_objects.append(manifest_obj)

        scene_hash = cls._stable_hash(
            {
                "version": 2,
                "config": config,
                "objects": manifest_objects,
            }
        )
        return (
            {
                "version": 2,
                "config": config,
                "objects": manifest_objects,
                "sceneHash": scene_hash,
            },
            assets,
        )

    @classmethod
    def _replace_visual_assets_with_refs(
        cls,
        obj,
        assets: dict[str, dict],
    ) -> None:
        for visual in obj.get("visuals") or []:
            mesh = visual.get("mesh")
            if mesh is not None:
                mesh_hash = cls._mesh_hash(mesh)
                mesh_asset = deepcopy(mesh)
                mesh_asset["hash"] = mesh_hash
                assets["meshes"][mesh_hash] = mesh_asset
                visual["mesh"] = {"hash": mesh_hash}

            material = visual.get("material")
            if material is not None:
                material_hash, material_asset, material_ref = cls._material_hash_and_asset(
                    material,
                    assets,
                )
                assets["materials"][material_hash] = material_asset
                visual["material"] = material_ref

    @classmethod
    def _mesh_hash(cls, mesh) -> str:
        return cls._stable_hash(
            {
                "indices": mesh.get("indices"),
                "vertices": mesh.get("vertices"),
                "normals": mesh.get("normals"),
                "uv": mesh.get("uv"),
            }
        )

    @classmethod
    def _texture_hash_and_asset(cls, texture) -> tuple[str, dict]:
        texture_hash = cls._stable_hash(
            {
                "width": texture.get("width"),
                "height": texture.get("height"),
                "textureType": texture.get("textureType"),
                "textureScale": texture.get("textureScale"),
                "textureData": texture.get("textureData"),
            }
        )
        texture_asset = deepcopy(texture)
        texture_asset["hash"] = texture_hash
        return texture_hash, texture_asset

    @classmethod
    def _material_hash_and_asset(
        cls,
        material,
        assets: dict[str, dict],
    ) -> tuple[str, dict, dict]:
        texture = material.get("texture")
        texture_ref = None
        if texture is not None:
            texture_hash, texture_asset = cls._texture_hash_and_asset(texture)
            assets["textures"][texture_hash] = texture_asset
            texture_ref = {"hash": texture_hash}

        material_for_hash = {
            "color": material.get("color"),
            "emissionColor": material.get("emissionColor"),
            "specular": material.get("specular"),
            "shininess": material.get("shininess"),
            "reflectance": material.get("reflectance"),
            "texture": texture_ref,
        }
        material_hash = cls._stable_hash(material_for_hash)
        material_asset = deepcopy(material)
        material_asset["hash"] = material_hash
        material_asset["texture"] = texture_ref
        material_ref = {"hash": material_hash}
        if texture_ref is not None:
            material_ref["texture"] = texture_ref
        return material_hash, material_asset, material_ref

    @classmethod
    def _object_content_hash(cls, obj) -> str:
        visuals = []
        for visual in obj.get("visuals") or []:
            mesh = visual.get("mesh")
            material = visual.get("material")
            visuals.append(
                {
                    "name": visual.get("name"),
                    "type": visual.get("type"),
                    "trans": visual.get("trans"),
                    "mesh": mesh.get("hash") if mesh is not None else None,
                    "material": material.get("hash") if material is not None else None,
                }
            )
        return cls._stable_hash(
            {
                "parent": obj.get("parent"),
                "visuals": visuals,
            }
        )

    def _pack_scene_manifest_payload(
        self,
    ) -> tuple[bytes, dict[str, dict[str, dict]], str, dict[str, int]]:
        payload, asset_store = self._build_scene_manifest_document(
            self._sim_scene.config,
            self._flat,
        )
        asset_counts = {
            kind: len(asset_store[kind])
            for kind in _ASSET_KINDS
        }
        return (
            msgpack.packb(payload, use_bin_type=True),
            asset_store,
            payload["sceneHash"],
            asset_counts,
        )

    async def _send_scene_manifest_raw(self):
        payload, asset_store, scene_hash, asset_counts = self._pack_scene_manifest_payload()
        self._asset_store = asset_store
        self._scene_hash = scene_hash
        await self._send_raw("scene_manifest", payload)

    async def _send_scene_manifest(self):
        try:
            await self._send_scene_manifest_raw()
        except websockets.ConnectionClosed as exc:
            code, reason, exc_type, detail = _connection_closed_details(exc)
            logger.warning(
                "Quest websocket send failed for msg_type='scene_manifest': code=%s reason=%r type=%s detail=%s",
                code,
                reason,
                exc_type,
                detail,
            )
            await self._drop_connection()
        except Exception:
            logger.exception("Quest websocket send failed for msg_type='scene_manifest'")
            await self._drop_connection()

    async def _connect(self):
        return await websockets.connect(
            f"ws://{self.quest_ip}:{self.quest_port}/sim",
            max_size=64 * 1024 * 1024,
            ping_interval=_PING_INTERVAL,
            ping_timeout=_PING_TIMEOUT,
            close_timeout=_CLOSE_TIMEOUT,
        )

    async def _send_raw(self, msg_type: str, data: bytes):
        """Send on the current socket, propagating any failure to the caller.

        Used by the reconnect path, which must know whether the scene resend on
        a fresh socket actually succeeded.
        """
        envelope = msgpack.packb({"type": msg_type, "data": data}, use_bin_type=True)
        await self._ws.send(envelope)

    async def _send(self, msg_type: str, data: bytes):
        try:
            await self._send_raw(msg_type, data)
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
            # Non-fatal: flag the drop and swallow. The supervisor (driven by
            # the recv loop ending) owns reconnection; raising here would crash
            # the caller's recording loop.
            await self._drop_connection()
        except Exception:
            logger.exception(
                "Quest websocket send failed for msg_type=%r",
                msg_type,
            )
            await self._drop_connection()

    def _build_asset_response(self, request: dict) -> dict:
        requested = {
            kind: [str(asset_hash) for asset_hash in (request.get(kind) or [])]
            for kind in _ASSET_KINDS
        }
        response = {
            "version": 1,
            "requestId": request.get("requestId"),
            "sceneHash": request.get("sceneHash"),
            "assets": self._empty_asset_store(),
            "missing": {kind: [] for kind in _ASSET_KINDS},
        }

        if request.get("sceneHash") != self._scene_hash:
            for kind in _ASSET_KINDS:
                response["missing"][kind].extend(requested[kind])
            logger.warning(
                "Ignoring stale asset_request requestId=%r sceneHash=%r currentSceneHash=%r",
                request.get("requestId"),
                request.get("sceneHash"),
                self._scene_hash,
            )
            return response

        for kind in _ASSET_KINDS:
            store = self._asset_store.get(kind, {})
            for asset_hash in requested[kind]:
                asset = store.get(asset_hash)
                if asset is None:
                    response["missing"][kind].append(asset_hash)
                else:
                    response["assets"][kind][asset_hash] = asset
        return response

    async def _handle_asset_request(self, payload: dict) -> None:
        response = self._build_asset_response(payload)
        await self._send(
            "asset_response",
            msgpack.packb(response, use_bin_type=True),
        )

    def _clear_input_state(self) -> None:
        with self._input_lock:
            self._latest_input = None
            for pending in self._pending.values():
                pending.clear()

    def _mark_disconnected(self) -> None:
        self._connected.clear()
        self._clear_input_state()

    async def _drop_connection(self) -> None:
        self._mark_disconnected()
        try:
            await self._ws.close()
        except Exception:
            pass

    async def _recv_loop(self):
        try:
            async for msg in self._ws:
                try:
                    envelope = msgpack.unpackb(msg, raw=False)
                except Exception:
                    continue
                msg_type = envelope.get("type")
                if msg_type == "input":
                    payload = msgpack.unpackb(envelope["data"], raw=False)
                    self._apply_input(payload)
                elif msg_type == "asset_request":
                    payload = msgpack.unpackb(envelope["data"], raw=False)
                    await self._handle_asset_request(payload)
        except websockets.ConnectionClosed as exc:
            code, reason, exc_type, detail = _connection_closed_details(exc)
            logger.warning(
                "Quest websocket receive loop closed: code=%s reason=%r type=%s detail=%s",
                code,
                reason,
                exc_type,
                detail,
            )
        except Exception:
            logger.exception("Quest websocket receive loop errored.")

    async def _supervise(self):
        """Own the connection lifecycle for the life of the publisher.

        Run the recv loop; when it returns (socket dropped), mark disconnected
        and reconnect with backoff. Reconnection lives here alone so that
        send-side and recv-side drops don't race two reconnect attempts.
        """
        while not self._closing:
            await self._recv_loop()
            if self._closing:
                break
            self._mark_disconnected()
            logger.warning("Quest connection lost; reconnecting…")
            await self._reconnect()

    async def _reconnect(self):
        """Reconnect forever with capped exponential backoff, then resend scene."""
        backoff = _INITIAL_BACKOFF
        while not self._closing:
            try:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = await self._connect()
                # Resend the current scene so a freshly (re)started Quest app
                # rebuilds it; refresh() keeps self._flat/_sim_scene current.
                # _send_raw so a failed resend retries instead of being
                # swallowed and falsely marked connected.
                await self._send_scene_manifest_raw()
                self._clear_input_state()
                self._connected.set()
                logger.info("Quest reconnected; scene resent.")
                return
            except Exception as exc:
                logger.warning(
                    "Quest reconnect attempt failed: %s; retrying in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    def is_connected(self) -> bool:
        return self._connected.is_set()

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
        if not self._connected.is_set():
            return
        payload = msgpack.packb(
            {"label": label, "value": str(value)}, use_bin_type=True
        )
        self._run(self._send("display", payload))

    def publish_state(self):
        # No-op while disconnected: skip the loop-thread round-trip entirely so
        # the caller's loop stays cheap (~20 Hz) until the supervisor reconnects.
        if not self._connected.is_set():
            return
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

    def rebind(self, env) -> None:
        """Re-bind to env's fresh mj_model/mj_data after a hard reset, no network I/O.

        robosuite's default ``hard_reset=True`` on ``env.reset()`` replaces
        ``env.sim`` (and thus the raw mj_model/mj_data) on every reset,
        orphaning the numpy views grabbed at construction time.  This is only
        for callers that intentionally want pose-only rebinding.
        """
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self._rebuild_tracked()

    def refresh(self, env, force: bool = False) -> None:
        """Resync after ``env.reset()``.

        Always send the complete logical scene manifest.  Asset bytes are served
        only in response to headset asset_request messages.
        """
        self.model = env.sim.model._model
        self.data = env.sim.data._data

        self._parse_scene()
        self._rebuild_tracked()
        # Only hit the network if connected; otherwise the supervisor's
        # reconnect will resend this now-current scene once the socket is back.
        if self._connected.is_set():
            self._run(self._send_scene_manifest())

    def close(self):
        # Stop the supervisor from reconnecting once we start tearing down.
        self._closing = True

        async def _close():
            try:
                await self._send("clear", b"")
            except Exception:
                pass
            await self._ws.close()

        try:
            self._supervisor_task.cancel()
        except Exception:
            pass
        try:
            self._run(_close())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2)

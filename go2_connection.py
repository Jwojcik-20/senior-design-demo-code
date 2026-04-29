from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import sys
from typing import List, Tuple

import requests
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.exceptions import InvalidStateError
from aiortc.rtcicetransport import RTCIceTransport
from aiortc.rtcdtlstransport import RTCDtlsTransport
from aiortc.rtcdtlstransport import RTCCertificate
import unitree_webrtc_connect.unitree_auth as unitree_auth
import unitree_webrtc_connect.webrtc_audio as unitree_webrtc_audio
import unitree_webrtc_connect.webrtc_datachannel as unitree_webrtc_datachannel
import unitree_webrtc_connect.webrtc_driver as unitree_webrtc_driver
import unitree_webrtc_connect.webrtc_video as unitree_webrtc_video
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod


GO2_IP = os.getenv("GO2_IP", "192.168.123.161").strip()
GO2_SERIAL = os.getenv("GO2_SERIAL", "").strip()
GO2_CONNECTION_MODE = os.getenv("GO2_CONNECTION_MODE", "auto").strip().lower()
GO2_WEBRTC_PORT_SETTING = os.getenv("GO2_WEBRTC_PORT", "auto").strip().lower()
GO2_HTTP_TIMEOUT = max(1.0, float(os.getenv("GO2_HTTP_TIMEOUT", "6")))
GO2_CONNECT_TIMEOUT = max(3.0, float(os.getenv("GO2_CONNECT_TIMEOUT", "15")))
GO2_DATACHANNEL_TIMEOUT = max(5.0, float(os.getenv("GO2_DATACHANNEL_TIMEOUT", "20")))
GO2_CONNECT_RETRIES = max(1, int(os.getenv("GO2_CONNECT_RETRIES", "3")))
UNITREE_EMAIL = os.getenv("UNITREE_EMAIL", "").strip()
UNITREE_PASS = os.getenv("UNITREE_PASS", "").strip()


def _ensure_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_ensure_utf8_stdio()

_PATCHED = False
_PATCH_LEVEL = "none"
_ORIGINAL_ICE_STOP = RTCIceTransport.stop
_ORIGINAL_DTLS_STOP = RTCDtlsTransport.stop
_ORIGINAL_SET_REMOTE_DESCRIPTION = RTCPeerConnection.setRemoteDescription
_ORIGINAL_PEER_CONNECT = RTCPeerConnection._RTCPeerConnection__connect
_ORIGINAL_WAIT_DATACHANNEL_OPEN = unitree_webrtc_datachannel.WebRTCDataChannel.wait_datachannel_open
_ORIGINAL_ADD_TRACK_CALLBACK = unitree_webrtc_video.WebRTCVideoChannel.add_track_callback
_ORIGINAL_VIDEO_TRACK_HANDLER = unitree_webrtc_video.WebRTCVideoChannel.track_handler
_ORIGINAL_GET_FINGERPRINTS = RTCCertificate.getFingerprints
_ORIGINAL_LOCAL_REQUEST = unitree_auth.make_local_request
_ORIGINAL_SEND_SDP_OLD = unitree_auth.send_sdp_to_local_peer_old_method
_ORIGINAL_SEND_SDP_NEW = unitree_auth.send_sdp_to_local_peer_new_method
_ORIGINAL_SEND_SDP = unitree_auth.send_sdp_to_local_peer
_ORIGINAL_DRIVER_SEND_SDP = unitree_webrtc_driver.send_sdp_to_local_peer


def _tcp_port_open(host: str, port: int, timeout: float = 0.75) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_local_signal_ports(ip: str | None = None) -> List[int]:
    raw = GO2_WEBRTC_PORT_SETTING
    if raw and raw != "auto":
        ports: List[int] = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                ports.append(int(chunk))
            except ValueError:
                continue
        if ports:
            return ports
    host = ip or GO2_IP
    preferred: List[int] = []
    for port in (9991, 8081):
        if _tcp_port_open(host, port):
            preferred.append(port)
    return preferred or [9991, 8081]


def _make_local_request_with_timeout(path, body=None, headers=None):
    try:
        response = requests.post(
            url=path,
            data=body,
            headers=headers,
            timeout=(GO2_HTTP_TIMEOUT, GO2_HTTP_TIMEOUT),
        )
        response.raise_for_status()
        return response if response.status_code == 200 else None
    except requests.exceptions.RequestException as exc:
        print(f"[WebRTC] POST {path} failed: {exc}")
        return None


def _send_sdp_offer_method(ip: str, sdp: str, port: int) -> str | None:
    url = f"http://{ip}:{port}/offer"
    response = _make_local_request_with_timeout(url, body=sdp, headers={"Content-Type": "application/json"})
    if response and response.status_code == 200:
        return response.text
    return None


def _send_sdp_new_method(ip: str, sdp: str, port: int = 9991) -> str | None:
    url = f"http://{ip}:{port}/con_notify"
    response = _make_local_request_with_timeout(url, body=None, headers=None)
    if not response:
        return None

    try:
        decoded_response = base64.b64decode(response.text).decode("utf-8")
        decoded_json = json.loads(decoded_response)
        data1 = decoded_json.get("data1")
        data2 = decoded_json.get("data2")
        if data2 == 2:
            data1 = unitree_auth.decrypt_con_notify_data(data1)
        public_key_pem = data1[10 : len(data1) - 10]
        path_ending = unitree_auth._calc_local_path_ending(data1)
        aes_key = unitree_auth.generate_aes_key()
        public_key = unitree_auth.rsa_load_public_key(public_key_pem)
        body = {
            "data1": unitree_auth.aes_encrypt(sdp, aes_key),
            "data2": unitree_auth.rsa_encrypt(aes_key, public_key),
        }
        url = f"http://{ip}:{port}/con_ing_{path_ending}"
        response = _make_local_request_with_timeout(
            url,
            body=json.dumps(body),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not response:
            return None
        return unitree_auth.aes_decrypt(response.text, aes_key)
    except Exception as exc:
        print(f"[WebRTC] 9991 encrypted handshake failed: {exc}")
        return None


def _send_sdp_to_local_peer(ip: str, sdp: str) -> str | None:
    ports = resolve_local_signal_ports(ip)
    for port in ports:
        if port == 9991:
            answer = _send_sdp_new_method(ip, sdp, port=port)
        else:
            answer = _send_sdp_offer_method(ip, sdp, port=port)
        if answer:
            print(f"[Go2] SDP exchange succeeded on port {port}")
            return answer
    print(f"[Go2] Local SDP exchange failed on ports: {', '.join(map(str, ports))}")
    return None


def patch_unitree_local_signaling(level: str = "full"):
    global _PATCHED, _PATCH_LEVEL
    level = (level or "full").strip().lower()
    if level not in {"signal", "full"}:
        raise ValueError(f"Unsupported patch level: {level}")

    async def _noop_ice_stop(self):
        return None

    async def _noop_dtls_stop(self):
        return None

    async def _patched_set_remote_description(self, sessionDescription):
        RTCIceTransport.stop = _noop_ice_stop
        RTCDtlsTransport.stop = _noop_dtls_stop
        try:
            return await _ORIGINAL_SET_REMOTE_DESCRIPTION(self, sessionDescription)
        finally:
            RTCIceTransport.stop = _ORIGINAL_ICE_STOP
            RTCDtlsTransport.stop = _ORIGINAL_DTLS_STOP

    async def _patched_connect(self):
        started_ice = set()
        started_dtls = set()
        remote_ice = self._RTCPeerConnection__remoteIce
        remote_dtls = self._RTCPeerConnection__remoteDtls

        for transceiver in self._RTCPeerConnection__transceivers:
            dtlsTransport = transceiver.receiver.transport
            iceTransport = dtlsTransport.transport
            if (
                iceTransport.iceGatherer.getLocalCandidates()
                and transceiver in remote_ice
                and iceTransport.state != "closed"
            ):
                if iceTransport not in started_ice:
                    try:
                        await iceTransport.start(remote_ice[transceiver])
                    except InvalidStateError:
                        continue
                    started_ice.add(iceTransport)
                if dtlsTransport.state == "new" and dtlsTransport not in started_dtls:
                    try:
                        await dtlsTransport.start(remote_dtls[transceiver])
                    except InvalidStateError:
                        continue
                    started_dtls.add(dtlsTransport)
                if dtlsTransport.state == "connected":
                    if transceiver.currentDirection in ["sendonly", "sendrecv"]:
                        await transceiver.sender.send(self._RTCPeerConnection__localRtp(transceiver))
                    if transceiver.currentDirection in ["recvonly", "sendrecv"]:
                        await transceiver.receiver.receive(self._RTCPeerConnection__remoteRtp(transceiver))

        if self._RTCPeerConnection__sctp:
            dtlsTransport = self._RTCPeerConnection__sctp.transport
            iceTransport = dtlsTransport.transport
            sctp = self._RTCPeerConnection__sctp
            if (
                iceTransport.iceGatherer.getLocalCandidates()
                and sctp in remote_ice
                and iceTransport.state != "closed"
            ):
                if iceTransport not in started_ice:
                    await iceTransport.start(remote_ice[sctp])
                    started_ice.add(iceTransport)
                if dtlsTransport.state == "new" and dtlsTransport not in started_dtls:
                    try:
                        await dtlsTransport.start(remote_dtls[sctp])
                    except InvalidStateError:
                        return
                    started_dtls.add(dtlsTransport)
                if dtlsTransport.state == "connected":
                    await sctp.start(self._RTCPeerConnection__sctpRemoteCaps, self._RTCPeerConnection__sctpRemotePort)

    async def _patched_wait_datachannel_open(self, timeout=5):
        return await _ORIGINAL_WAIT_DATACHANNEL_OPEN(self, timeout=max(timeout, GO2_DATACHANNEL_TIMEOUT))

    def _patched_get_fingerprints(self):
        fingerprints = _ORIGINAL_GET_FINGERPRINTS(self)
        sha256_only = [fp for fp in fingerprints if getattr(fp, "algorithm", None) == "sha-256"]
        return sha256_only or fingerprints

    def _patched_add_track_callback(self, callback):
        _ORIGINAL_ADD_TRACK_CALLBACK(self, callback)
        pending_track = getattr(self, "_pending_video_track", None)
        if pending_track is None or not callable(callback):
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(callback(pending_track))
        except RuntimeError:
            pass

    async def _patched_video_track_handler(self, track):
        self._pending_video_track = track
        if not self.track_callbacks:
            logging.info("Video track received before callback registration; holding track for later callback.")
            return
        await _ORIGINAL_VIDEO_TRACK_HANDLER(self, track)

    RTCPeerConnection.setRemoteDescription = _ORIGINAL_SET_REMOTE_DESCRIPTION
    RTCPeerConnection._RTCPeerConnection__connect = _ORIGINAL_PEER_CONNECT
    RTCCertificate.getFingerprints = _ORIGINAL_GET_FINGERPRINTS
    unitree_webrtc_datachannel.WebRTCDataChannel.wait_datachannel_open = _ORIGINAL_WAIT_DATACHANNEL_OPEN
    unitree_webrtc_video.WebRTCVideoChannel.add_track_callback = _ORIGINAL_ADD_TRACK_CALLBACK
    unitree_webrtc_video.WebRTCVideoChannel.track_handler = _ORIGINAL_VIDEO_TRACK_HANDLER
    unitree_auth.make_local_request = _ORIGINAL_LOCAL_REQUEST
    unitree_auth.send_sdp_to_local_peer_old_method = _ORIGINAL_SEND_SDP_OLD
    unitree_auth.send_sdp_to_local_peer_new_method = _ORIGINAL_SEND_SDP_NEW
    unitree_auth.send_sdp_to_local_peer = _ORIGINAL_SEND_SDP
    unitree_webrtc_driver.send_sdp_to_local_peer = _ORIGINAL_DRIVER_SEND_SDP

    if level == "full":
        RTCPeerConnection.setRemoteDescription = _patched_set_remote_description
        RTCPeerConnection._RTCPeerConnection__connect = _patched_connect
        RTCCertificate.getFingerprints = _patched_get_fingerprints
        unitree_webrtc_datachannel.WebRTCDataChannel.wait_datachannel_open = _patched_wait_datachannel_open
        unitree_webrtc_video.WebRTCVideoChannel.add_track_callback = _patched_add_track_callback
        unitree_webrtc_video.WebRTCVideoChannel.track_handler = _patched_video_track_handler

    unitree_auth.make_local_request = _make_local_request_with_timeout
    unitree_auth.send_sdp_to_local_peer_old_method = _send_sdp_to_local_peer
    unitree_auth.send_sdp_to_local_peer_new_method = _send_sdp_new_method
    unitree_auth.send_sdp_to_local_peer = _send_sdp_to_local_peer
    unitree_webrtc_driver.send_sdp_to_local_peer = _send_sdp_to_local_peer
    _PATCHED = True
    _PATCH_LEVEL = level


async def start_go2_video_stream(
    conn: UnitreeWebRTCConnection,
    track_callback,
    *,
    first_frame_timeout: float = 15.0,
):
    first_frame = asyncio.Event()

    async def _wrapped(track):
        if not first_frame.is_set():
            first_frame.set()
        await track_callback(track)

    conn.video.add_track_callback(_wrapped)
    conn.video.switchVideoChannel(True)
    await asyncio.wait_for(first_frame.wait(), timeout=first_frame_timeout)


async def connect_go2_video_only(
    track_callback,
    ip: str | None = None,
    timeout: float | None = None,
) -> RTCPeerConnection:
    patch_unitree_local_signaling()
    host = (ip or GO2_IP).strip()
    limit = timeout or GO2_CONNECT_TIMEOUT
    pc = RTCPeerConnection()
    ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track):
        if track.kind != "video":
            return
        ready.set()
        await track_callback(track)

    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    payload = {
        "id": "STA_localNetwork",
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "token": "",
    }
    answer_text = _send_sdp_to_local_peer(host, json.dumps(payload))
    if not answer_text:
        await pc.close()
        raise RuntimeError("Go2 returned no SDP answer for video-only connection.")

    answer = json.loads(answer_text)
    await asyncio.wait_for(
        pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])),
        timeout=limit,
    )
    await asyncio.wait_for(ready.wait(), timeout=limit)
    print("[Go2] Video-only WebRTC connected", flush=True)
    return pc


async def connect_go2_media_only(
    track_callback,
    ip: str | None = None,
    timeout: float | None = None,
    include_audio: bool = True,
) -> RTCPeerConnection:
    patch_unitree_local_signaling("full")
    host = (ip or GO2_IP).strip()
    limit = timeout or GO2_CONNECT_TIMEOUT
    pc = RTCPeerConnection()
    ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            ready.set()
            await track_callback(track)
            return
        if track.kind == "audio":
            return

    if include_audio:
        pc.addTransceiver("audio", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    payload = {
        "id": "STA_localNetwork",
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "token": "",
    }
    answer_text = _send_sdp_to_local_peer(host, json.dumps(payload))
    if not answer_text:
        await pc.close()
        raise RuntimeError("Go2 returned no SDP answer for media-only connection.")

    answer = json.loads(answer_text)
    await asyncio.wait_for(
        pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])),
        timeout=limit,
    )
    await asyncio.wait_for(ready.wait(), timeout=limit)
    print("[Go2] Media-only WebRTC connected", flush=True)
    return pc


async def connect_go2_control_only(
    ip: str | None = None,
    timeout: float | None = None,
):
    patch_unitree_local_signaling()
    host = (ip or GO2_IP).strip()
    limit = timeout or GO2_CONNECT_TIMEOUT
    pc = RTCPeerConnection()
    stub = type("Go2ControlOnlyStub", (), {})()
    datachannel = unitree_webrtc_datachannel.WebRTCDataChannel(stub, pc)

    payload = {
        "id": "STA_localNetwork",
        "sdp": None,
        "type": None,
        "token": "",
    }
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    payload["sdp"] = pc.localDescription.sdp
    payload["type"] = pc.localDescription.type
    answer_text = _send_sdp_to_local_peer(host, json.dumps(payload))
    if not answer_text:
        await pc.close()
        raise RuntimeError("Go2 returned no SDP answer for control-only connection.")

    answer = json.loads(answer_text)
    await asyncio.wait_for(
        pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])),
        timeout=limit,
    )
    await asyncio.wait_for(datachannel.wait_datachannel_open(timeout=limit), timeout=limit + 1.0)
    print("[Go2] Control-only WebRTC connected", flush=True)
    return pc, datachannel


async def connect_go2_single_peer_camera(
    track_callback,
    ip: str | None = None,
    timeout: float | None = None,
    patch_level: str = "signal",
) -> UnitreeWebRTCConnection:
    patch_unitree_local_signaling(patch_level)
    host = (ip or GO2_IP).strip()
    limit = timeout or GO2_CONNECT_TIMEOUT
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=host)
    pc = RTCPeerConnection()
    conn.pc = pc
    conn.datachannel = unitree_webrtc_datachannel.WebRTCDataChannel(conn, pc)
    conn.audio = unitree_webrtc_audio.WebRTCAudioChannel(pc, conn.datachannel)
    conn.video = unitree_webrtc_video.WebRTCVideoChannel(pc, conn.datachannel)
    conn.video.add_track_callback(track_callback)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            await conn.video.track_handler(track)
            return
        if track.kind == "audio":
            return

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    payload = {
        "id": "STA_localNetwork",
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "token": conn.token,
    }
    answer_text = _send_sdp_to_local_peer(host, json.dumps(payload))
    if not answer_text:
        await pc.close()
        raise RuntimeError("Go2 returned no SDP answer for single-peer camera connection.")

    answer = json.loads(answer_text)
    await asyncio.wait_for(
        pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])),
        timeout=limit,
    )

    async def _enable_video():
        while not conn.datachannel.data_channel_opened:
            await asyncio.sleep(0.1)
        try:
            await conn.datachannel.disableTrafficSaving(True)
        except Exception:
            pass
        conn.video.switchVideoChannel(True)

    await asyncio.wait_for(_enable_video(), timeout=limit)
    print("[Go2] Single-peer camera WebRTC connected", flush=True)
    return conn


def _connection_mode_candidates(
    mode: str | None = None,
    ip: str | None = None,
    serial: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> List[Tuple[str, WebRTCConnectionMethod, dict]]:
    resolved_mode = (mode or GO2_CONNECTION_MODE or "auto").strip().lower()
    resolved_ip = (ip or GO2_IP).strip()
    resolved_serial = (serial or GO2_SERIAL).strip()
    resolved_user = username if username is not None else UNITREE_EMAIL
    resolved_pass = password if password is not None else UNITREE_PASS

    local_kwargs = {}
    if resolved_ip:
        local_kwargs["ip"] = resolved_ip
    elif resolved_serial:
        local_kwargs["serialNumber"] = resolved_serial
    if resolved_user and resolved_pass:
        local_kwargs["username"] = resolved_user
        local_kwargs["password"] = resolved_pass

    remote_kwargs = {}
    if resolved_serial and resolved_user and resolved_pass:
        remote_kwargs = {
            "serialNumber": resolved_serial,
            "username": resolved_user,
            "password": resolved_pass,
        }

    if resolved_mode == "localap":
        return [("LocalAP", WebRTCConnectionMethod.LocalAP, {})]
    if resolved_mode == "localsta":
        return [("LocalSTA", WebRTCConnectionMethod.LocalSTA, local_kwargs)]
    if resolved_mode == "remote":
        return [("Remote", WebRTCConnectionMethod.Remote, remote_kwargs)]

    candidates: List[Tuple[str, WebRTCConnectionMethod, dict]] = []
    if local_kwargs:
        candidates.append(("LocalSTA", WebRTCConnectionMethod.LocalSTA, local_kwargs))
    if "ip" not in local_kwargs:
        candidates.append(("LocalAP", WebRTCConnectionMethod.LocalAP, {}))
    if remote_kwargs:
        candidates.append(("Remote", WebRTCConnectionMethod.Remote, remote_kwargs))
    return candidates


async def connect_best_go2(
    mode: str | None = None,
    ip: str | None = None,
    serial: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> tuple[UnitreeWebRTCConnection, str]:
    patch_unitree_local_signaling()
    candidates = _connection_mode_candidates(
        mode=mode,
        ip=ip,
        serial=serial,
        username=username,
        password=password,
    )
    if not candidates:
        raise RuntimeError(
            "No Go2 connection candidates available. Set GO2_IP for local use, or set "
            "GO2_SERIAL + UNITREE_EMAIL + UNITREE_PASS for remote use."
        )

    errors: List[str] = []
    for label, method, kwargs in candidates:
        for attempt in range(1, GO2_CONNECT_RETRIES + 1):
            conn = None
            task = None
            connected = False
            try:
                suffix = f" (attempt {attempt}/{GO2_CONNECT_RETRIES})" if GO2_CONNECT_RETRIES > 1 else ""
                print(f"[Go2] Trying {label} connection{suffix}...", flush=True)
                conn = UnitreeWebRTCConnection(method, **kwargs)
                task = asyncio.create_task(conn.connect())
                await asyncio.wait_for(task, timeout=GO2_CONNECT_TIMEOUT)
                connected = True
                print(f"[Go2] Connected via {label}", flush=True)
                return conn, label
            except SystemExit as exc:
                errors.append(f"{label} attempt {attempt}: connect aborted ({exc})")
            except asyncio.TimeoutError:
                errors.append(f"{label} attempt {attempt}: timed out after {GO2_CONNECT_TIMEOUT:.1f}s")
            except Exception as exc:
                errors.append(f"{label} attempt {attempt}: {exc}")
            finally:
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)
                    except Exception:
                        pass
                if conn is not None and not connected:
                    try:
                        await asyncio.wait_for(conn.disconnect(), timeout=2.0)
                    except Exception:
                        pass
            await asyncio.sleep(0.3)

    joined = "\n".join(f"- {item}" for item in errors)
    raise RuntimeError(f"All Go2 connection attempts failed:\n{joined}")

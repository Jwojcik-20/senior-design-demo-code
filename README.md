# Senior Design Demo Code

Senior design workspace for a Unitree Go2 demo that combines robot control, waypoint navigation, live mapping, and face-recognition-driven live video workflows.

## What is in this repo

- `go2_control_center.py`: main desktop launcher that combines motion control, waypoints, live map, and embedding tools in one Tkinter app
- `go2_motion_ui.py`: standalone motion control panel
- `go2_waypoint_ui.py`: standalone waypoint manager
- `go2_live_map_ui.py`: standalone live map viewer
- `embedding_protocol_ui.py`: desktop workflow for collecting faces, training embeddings, and running recognition
- `display_video_channel_with_faces_login.py`: live Go2 interface with login screen, recognition overlays, alert clips, and optional robot reactions
- `embedding_face_recognition_dual_display.py`: embedding-based face collection, training, and live recognition backend
- `go2_connection.py`: shared connection and signaling helpers for local and remote Go2 access
- `go2_navigation.py`: waypoint and closed-loop navigation backend
- `go2_camera_probe.py`: connection and camera diagnostics
- `test_go2_motion.py`: direct motion command runner for quick robot command tests

## Requirements

- Windows machine with Python 3.12 recommended
- Unitree Go2 reachable over local network or through Unitree account credentials
- OpenCV-compatible camera and display environment
- `ffmpeg` installed if you want alert clip export
- Optional audio support through `PyAudio`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` values into your shell or a local `.env` workflow of your choice.
4. Set the Go2 connection variables for your robot.
5. Run one of the entry points below.

Common entry points:

```bash
python go2_control_center.py
python embedding_protocol_ui.py
python display_video_channel_with_faces_login.py
python test_go2_motion.py StandUp
```

## Configuration

The scripts are driven mostly by environment variables.

Core Go2 connection settings:

- `GO2_IP`: local IP of the robot for local-network access
- `GO2_CONNECTION_MODE`: connection strategy such as `auto`, `local`, or `hybrid`
- `GO2_SERIAL`: robot serial for remote Unitree flows
- `UNITREE_EMAIL`: Unitree account email
- `UNITREE_PASS`: Unitree account password

SSH fallback settings used by the embedding workflow:

- `GO2_SSH_HOST`
- `GO2_SSH_USER`
- `GO2_SSH_PASS`
- `GO2_SSH_REMOTE_GLOB`
- `GO2_SSH_VIDEO_CLIENT`

Live interface settings:

- `LIVEINTERFACE_SOURCE`
- `LIVEINTERFACE_USER`
- `LIVEINTERFACE_PASS`
- `FACE_BACKEND`
- `GO2_REACT_ENABLED`
- `GO2_STAND_ON_RECOGNITION`
- `GO2_SIT_ON_DETECT`
- `ALERT_EMAIL_SENDER`
- `ALERT_EMAIL_PASSWORD`
- `ALERT_EMAIL_RECIPIENTS`

See `.env.example` for a practical starter set.

## Tracked assets

This repository intentionally keeps the code and core OpenCV model files in git while excluding local runtime artifacts such as:

- virtual environments
- collected datasets
- generated face embeddings and labels
- local SSH cache and WebRTC bridge cache
- logs and alert clips

The current `.gitignore` already excludes those machine-specific files.

## Notes

- `models/opencv_zoo/` contains third-party model files used by the face-recognition pipeline.
- Runtime-generated model artifacts such as `lbph_face.yml` and local embeddings are ignored so personal training data is not published accidentally.
- Tkinter is used for the desktop UI and is normally bundled with standard Python on Windows.

## License

The source code in this repository is licensed under the MIT License. Third-party assets and model files included from upstream projects remain under their own licenses.

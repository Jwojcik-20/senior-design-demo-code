# Senior Design Demo Code

Go2 robot control and perception workspace.

## Main apps

- `go2_control_center.py`: tabbed control center for motion, waypoints, live map, and embedding tools
- `go2_motion_ui.py`: standalone motion control UI
- `go2_waypoint_ui.py`: standalone waypoint manager
- `go2_live_map_ui.py`: standalone live map viewer
- `embedding_protocol_ui.py`: standalone embedding workflow UI
- `display_video_channel_with_faces_login.py`: Go2 WebRTC live interface using the embedding backend

## Robot-side helpers

- `test_go2_motion.py`: direct motion command runner
- `go2_navigation.py`: waypoint and closed-loop navigation backend
- `go2_connection.py`: shared Go2 connection helpers
- `go2_camera_probe.py`: camera probe / diagnostics

## Notes

- Local virtual environments, logs, training images, and generated artifacts are ignored by git.
- Go2 WebRTC and SSH settings are controlled through environment variables used by the scripts.

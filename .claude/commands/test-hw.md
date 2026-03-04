Run hardware diagnostic tests for YAM bimanual setup.

Check CAN interfaces, cameras, and input devices.

Steps:
1. Check CAN interfaces are up:
   ```bash
   ip link show | grep can
   ```
   Expected: can_follow_l and can_follow_r should be UP

2. Test RealSense cameras:
   ```bash
   uv run scripts/test_realsense_cameras.py
   ```

3. Test GELLO (Dynamixel leader arms):
   ```bash
   uv run scripts/test_gello_input.py
   ```

4. Test VR input (requires XRoboToolkit service):
   ```bash
   uv run scripts/test_vr_input.py
   ```

Run whichever diagnostics are relevant to the user's issue.
Report any failures clearly with suggestions to fix.

# -*- coding: utf-8 -*-
#
#  ...........       ____  *_*
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / **  / / **/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
# MIT License
# Copyright (c) 2023 Bitcraze
"""
file: crazyflie_py_wallfollowing.py
Controls the crazyflie and implements a wall following method in webots in Python
Author:   Kimberly McGuire (Bitcraze AB)
"""
from controller import Robot
from controller import Keyboard
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller
from object_detection import HumanDetector
from gpr_detection import GPRDetector
from gpr_bscan_plot import GPRBScanVisualizer
import time
import signal
import sys
from gpr_detection import WebotsGPRInterface
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
from autonomous_navigator import AdvancedAutonomousNavigator

FLYING_ATTITUDE = 1
SCAN_INTERVAL = 1.0  # GPR scan every 1.0 seconds
MIN_SCAN_DISTANCE = 0.2  # Minimum distance to move before next scan

# B-scan configuration
BSCAN_AUTO_GENERATE = True  # Auto-generate B-scans during flight
BSCAN_SAVE_RAW_DATA = True  # Save raw scan data
BSCAN_QUALITY = 'high'      # 'low', 'medium', 'high'
MAX_SCANS_PER_BSCAN = 50    # Limit scans per B-scan to manage memory
BSCAN_UPDATE_INTERVAL = 10  # Update live B-scan every 10 scans

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down gracefully...')
    if 'gpr' in globals():
        gpr.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)
    
    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    DISASTER_AREA_BOUNDS = (-10, 30, -10, 30)  # Adjust based on your world size
    navigator = AdvancedAutonomousNavigator(gps, world_bounds=DISASTER_AREA_BOUNDS)

    
    # Initialize detection systems
    detector = HumanDetector(camera)
    
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    # Flight data logging
    altitude_log = []
    time_log = []
    
    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    class GPSControllerAdapter:
        def __init__(self, gps_device):
            self.gps = gps_device

        def get_position(self):
            pos = self.gps.getValues()
            return (pos[0], pos[1], pos[2])

    # Initialize GPR detector
    try:
        # Use relative paths that work with your project structure
        template_path = "gpr_models/template_human.in"  # Remove ../../ 
        output_dir = "gpr_models/outputs"  # Remove ../../

        gpr_detector = GPRDetector(
            gpr_template_path=template_path,
            gpr_output_dir=output_dir,
            threshold=0.05
        )

        gpr = WebotsGPRInterface(
            gpr_detector=gpr_detector,
            robot_controller=GPSControllerAdapter(gps),
        )
        print("[INFO] GPR detector initialized successfully")
        print(f"[INFO] Simulating {len(gpr_detector.simulator.buried_objects)} buried objects")
        gpr_available = True
    except Exception as e:
        print(f"[ERROR] Failed to initialize GPR detector: {e}")
        gpr = None
        gpr_available = False

    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # GPR scanning variables
    last_scan_time = 0
    last_scan_x = 0
    last_scan_y = 0
    scan_counter = 0

    # B-scan control variables
    live_bscan_enabled = False
    bscan_batch_counter = 0


    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()
    height_desired = FLYING_ATTITUDE

    autonomous_mode = False
    gpr_scanning_enabled = True

    print("\n")
    print("====== Controls =======")
    print(" The Crazyflie can be controlled from your keyboard!")
    print(" All controllable movement is in body coordinates")
    print("- Use the up, back, right and left button to move in the horizontal plane")
    print("- Use Q and E to rotate around yaw")
    print("- Use W and S to go up and down")
    print("- Press A to start autonomous mode")
    print("- Press D to disable autonomous mode")
    print("- Press G to toggle GPR scanning")
    print("- Press R to generate GPR summary report")
    print("- Press B to generate B-scan visualization")
    print("- Press M to generate detection map")
    print("- Press T to generate 3D visualization")
    print("- Press H to run human body detection analysis")
    print("- Press P to generate publication-quality B-scan plots")
    print("- Press L to toggle live B-scan updates")
    print("- Press X to export raw scan data")
    print("- Press N to force navigation replanning")
    print("- Press C to show coverage statistics")

    # Main loop:
    try:
        while robot.step(timestep) != -1:
            current_time = robot.getTime()
            dt = current_time - past_time

            if first_time:
                past_x_global = gps.getValues()[0]
                past_y_global = gps.getValues()[1]
                past_time = current_time
                first_time = False
                continue

            # Get sensor data
            roll = imu.getRollPitchYaw()[0]
            pitch = imu.getRollPitchYaw()[1]
            yaw = imu.getRollPitchYaw()[2]
            yaw_rate = gyro.getValues()[2]
            x_global = gps.getValues()[0]
            v_x_global = (x_global - past_x_global)/dt if dt > 0 else 0
            y_global = gps.getValues()[1]
            v_y_global = (y_global - past_y_global)/dt if dt > 0 else 0
            altitude = gps.getValues()[2]

            # Log altitude and time data
            altitude_log.append(altitude)
            time_log.append(current_time)

            # Get body fixed velocities
            cos_yaw = cos(yaw)
            sin_yaw = sin(yaw)
            v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
            v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

            # Initialize control values
            forward_desired = 0
            sideways_desired = 0
            yaw_desired = 0
            height_diff_desired = 0

            # Handle keyboard input
            key = keyboard.getKey()
            while key > 0:
                if key == Keyboard.UP:
                    forward_desired += 0.5
                elif key == Keyboard.DOWN:
                    forward_desired -= 0.5
                elif key == Keyboard.RIGHT:
                    sideways_desired -= 0.5
                elif key == Keyboard.LEFT:
                    sideways_desired += 0.5
                elif key == ord('Q'):
                    yaw_desired = +1
                elif key == ord('E'):
                    yaw_desired = -1
                elif key == ord('W'):
                    height_diff_desired = 0.1
                elif key == ord('S'):
                    height_diff_desired = -0.1
                elif key == ord('A'):
                    if autonomous_mode is False:
                        autonomous_mode = True
                        print("Autonomous mode: ON")
                elif key == ord('D'):
                    if autonomous_mode is True:
                        autonomous_mode = False
                        print("Autonomous mode: OFF")
                elif key == ord('G'):
                    gpr_scanning_enabled = not gpr_scanning_enabled
                    print(f"GPR scanning: {'ON' if gpr_scanning_enabled else 'OFF'}")
                elif key == ord('R'):
                    if gpr:
                        summary = gpr.get_detection_summary()
                        print(f"GPR Summary: {summary}")
                elif key == ord('B'):
                    if gpr and gpr.scan_log:
                        print("[INFO] Generating B-scan visualization...")
                        try:
                            bscan = GPRBScanVisualizer(gpr.scan_log, gpr.gpr_detector)
                            bscan.generate_bscan_image()
                            bscan.generate_comprehensive_report()
                        
                            print("[INFO] B-scan analysis complete")
                        except Exception as e:
                            print(f"[WARNING] Failed to generate B-scan: {e}")
                elif key == ord('M'):
                    if gpr:
                        print("[INFO] Generating detection map...")
                        gpr.plot_detection_map()
                elif key == ord('T'):
                    if gpr:
                        print("[INFO] Generating 3D visualization...")
                        try:
                            bscan = GPRBScanVisualizer(gpr.scan_log, gpr.gpr_detector)
                            bscan.create_3d_visualization()
                            print("[WARNING] 3D visualization not implemented.")
                        except Exception as e:
                            print(f"[WARNING] Failed to generate 3D visualization: {e}")
                elif key == ord('H'):  # Add this new key for human-specific detection
                    if gpr and gpr.scan_log:
                        print("[INFO] Running human body detection analysis...")
                        try:
                            bscan = GPRBScanVisualizer(gpr.scan_log, gpr.gpr_detector)
                            human_detections = bscan.generate_bscan_with_human_detection()

                            if human_detections:
                                print(f"[ALERT] {len(human_detections)} potential human bodies detected!")
                                for i, det in enumerate(human_detections):
                                    print(f"  Body {i+1}: {det['length_meters']:.1f}m × {det['width_m']:.1f}m "
                                          f"(confidence: {det['confidence']:.3f})")
                            else:
                                print("[INFO] No human body signatures detected")
                        except Exception as e:
                            print(f"[WARNING] Human detection analysis failed: {e}")
                elif key == ord('N'):  # Force navigation replan
                    if autonomous_mode:
                        navigator.force_replan()
                        print("Navigation replanning forced")
                elif key == ord('C'):  # Show coverage statistics
                    progress = navigator.get_coverage_progress()
                    print(f"=== COVERAGE STATISTICS ===")
                    print(f"Waypoints: {progress['completed_waypoints']}/{progress['total_waypoints']}")
                    print(f"Progress: {progress['progress_percentage']:.1f}%")
                    print(f"Grid Coverage: {progress['grid_coverage']:.1f}%")
                    print(f"Current State: {progress['current_state']}")
                    if progress['current_target']:
                        print(f"Current Target: ({progress['current_target'][0]:.1f}, {progress['current_target'][1]:.1f})") 
                key = keyboard.getKey()

            height_desired += height_diff_desired * dt

            # Human detection (visual)
            detector.detect_async()
            detections = detector.get_latest_detections()
            detector.display_detections(detections)

            human_detected = False
            if len(detections) > 0:
                human_detected = True
                #print(f"[ALERT] {len(detections)} human(s) detected at current position!")
                # Log detection with GPS coordinates for rescue teams
                current_pos = gps.getValues()
                #print(f"[RESCUE] Human detected at coordinates: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
            # GPR scanning logic
            if gpr_available and gpr_scanning_enabled and not human_detected:
                # Check if enough time has passed and we've moved enough distance
                time_since_last_scan = current_time - last_scan_time
                distance_moved = ((x_global - last_scan_x)**2 + (y_global - last_scan_y)**2)**0.5

                if (time_since_last_scan >= SCAN_INTERVAL and 
                    distance_moved >= MIN_SCAN_DISTANCE):

                    try:
                        scan_label = f"scan_{scan_counter:03d}"
                        success = gpr.scan_position(x_global, y_global, altitude - 0.1, scan_label)

                        if success:
                            last_scan_time = current_time
                            last_scan_x = x_global
                            last_scan_y = y_global
                            scan_counter += 1
                            print(f"[INFO] GPR scan {scan_counter} at ({x_global:.2f}, {y_global:.2f}) - "
                                  f"Checking for buried objects...")

                            # Check for immediate detection feedback
                            latest_result = gpr.get_latest_detection()
                            if latest_result and latest_result.has_detection:
                                print(f"[ALERT] Buried object detected! Confidence: {latest_result.detection_confidence:.3f}")

                            # Live B-scan updates
                            if (live_bscan_enabled and BSCAN_AUTO_GENERATE and 
                                scan_counter % BSCAN_UPDATE_INTERVAL == 0 and 
                                len(gpr.scan_log) > 5):
                                
                                try:
                                    print(f"[INFO] Generating live B-scan update (scan {scan_counter})...")
                                    bscan = GPRBScanVisualizer(gpr.scan_log, gpr.gpr_detector)
                                    bscan.generate_bscan_image(save_name=f"live_bscan_{scan_counter:03d}.png")
                                    print("[INFO] Live B-scan updated")
                                except Exception as e:
                                    print(f"[WARNING] Live B-scan update failed: {e}")

                        else:
                            print(f"[WARNING] Failed to queue GPR scan at ({x_global:.2f}, {y_global:.2f})")                   
                    except Exception as e:
                        print(f"[ERROR] Failed to queue GPR scan: {e}")
            # Get range measurements in meters
            range_measurements = {
                'front': range_front.getValue() / 1000,
                'left': range_left.getValue() / 1000,
                'back': range_back.getValue() / 1000,
                'right': range_right.getValue() / 1000
            }
            # Update obstacle map and get navigation commands
            current_position = (gps.getValues()[0], gps.getValues()[1])
            navigator.update_obstacle_map(current_position, range_measurements, detection_threshold=1.5)
            
            # Apply autonomous control if enabled
            if autonomous_mode:
                # Get navigation commands from advanced navigator
                nav_forward, nav_sideways, nav_height_adjust = navigator.compute_navigation_control(
                    current_position, current_time
                )
                
                forward_desired += nav_forward
                sideways_desired += nav_sideways
                height_diff_desired += nav_height_adjust
                
                # Update target flight height
                target_height = navigator.get_current_flight_height()
                if target_height != FLYING_ATTITUDE:
                    height_desired = target_height
                
                # Print progress information periodically
                if int(current_time) % 10 == 0:  # Every 10 seconds
                    progress = navigator.get_coverage_progress()
                    print(f"[PROGRESS] Waypoint {progress['completed_waypoints']}/{progress['total_waypoints']} "
                          f"({progress['progress_percentage']:.1f}%) - Grid coverage: {progress['grid_coverage']:.1f}%")
                
                # Check if mission is complete
                if navigator.is_mission_complete():
                    print("[MISSION] Area coverage complete! Landing...")
                    autonomous_mode = False
            

            # PID velocity controller with fixed height
            motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                            yaw_desired, height_desired,
                                            roll, pitch, yaw_rate,
                                            altitude, v_x, v_y)

            # Apply motor commands
            m1_motor.setVelocity(-motor_power[0])
            m2_motor.setVelocity(motor_power[1])
            m3_motor.setVelocity(-motor_power[2])
            m4_motor.setVelocity(motor_power[3])

            # Update previous values
            past_time = current_time
            past_x_global = x_global
            past_y_global = y_global

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Cleanup and generate reports
        print("\n[INFO] Shutting down and generating reports...")

        # Define the plotting function outside the conditional
        def plot_altitude_vs_time(time_data, altitude_data, save_path="flight_altitude_profile.png"):
            """Plot altitude vs time graph for the drone flight"""
            try:
                plt.figure(figsize=(12, 8))

                # Main altitude plot
                plt.subplot(2, 1, 1)
                plt.plot(time_data, altitude_data, 'b-', linewidth=2, label='Drone Altitude')
                plt.axhline(y=FLYING_ATTITUDE, color='r', linestyle='--', 
                   label=f'Target Altitude ({FLYING_ATTITUDE}m)', alpha=0.7)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Altitude (m)')
                plt.title('Drone Flight Profile - Altitude vs Time')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Add statistics
                if len(altitude_data) > 0:
                    avg_altitude = sum(altitude_data) / len(altitude_data)
                    max_altitude = max(altitude_data)
                    min_altitude = min(altitude_data)

                    plt.text(0.02, 0.98, 
                            f'Flight Statistics:\n'
                            f'Duration: {time_data[-1] - time_data[0]:.1f}s\n'
                            f'Avg Altitude: {avg_altitude:.2f}m\n'
                            f'Max Altitude: {max_altitude:.2f}m\n'
                            f'Min Altitude: {min_altitude:.2f}m\n'
                            f'Target: {FLYING_ATTITUDE}m',
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Altitude deviation plot
                plt.subplot(2, 1, 2)
                altitude_deviation = [alt - FLYING_ATTITUDE for alt in altitude_data]
                plt.plot(time_data, altitude_deviation, 'g-', linewidth=1.5, label='Altitude Deviation')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.5, label='Perfect Tracking')
                plt.fill_between(time_data, altitude_deviation, alpha=0.3, color='green')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Deviation from Target (m)')
                plt.title('Altitude Control Performance')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[INFO] Altitude profile saved to {save_path}")
                plt.show()

            except Exception as e:
                print(f"[ERROR] Failed to generate altitude plot: {e}")

        # Generate altitude profile if data is available
        if len(altitude_log) > 0 and len(time_log) > 0:
            print("[INFO] Generating altitude vs time graph...")
            plot_altitude_vs_time(time_log, altitude_log)

        if gpr:
            try:
                # Wait for any pending scans to complete
                time.sleep(2)

                # Generate comprehensive reports
                summary = gpr.get_detection_summary()
                print(f"\n=== GPR MISSION SUMMARY ===")
                print(f"Total scans: {summary['total_scans']}")
                print(f"Detections: {summary['detections']}")
                print(f"Detection rate: {summary['detection_rate']:.2%}")
                print(f"Max amplitude: {summary['max_amplitude']:.6f}")

                if gpr.scan_log:
                    print("[INFO] Generating comprehensive GPR analysis...")
                    # Process in batches to avoid memory issues if needed
                    total_scans = len(gpr.scan_log)
                    if total_scans > MAX_SCANS_PER_BSCAN:
                        print(f"[INFO] Processing {total_scans} scans in batches of {MAX_SCANS_PER_BSCAN}...")
                        for batch_start in range(0, total_scans, MAX_SCANS_PER_BSCAN):
                            batch_end = min(batch_start + MAX_SCANS_PER_BSCAN, total_scans)
                            batch_data = gpr.scan_log[batch_start:batch_end]
                            
                            try:
                                print(f"[INFO] Processing batch {batch_start//MAX_SCANS_PER_BSCAN + 1}...")
                                bscan_batch = GPRBScanVisualizer(batch_data, gpr.gpr_detector)
                                bscan_batch.generate_bscan_image(
                                    save_name=f"batch_bscan_{batch_start:03d}_{batch_end:03d}.png"
                                )
                                if BSCAN_SAVE_RAW_DATA:
                                    bscan_batch.export_data_for_external_processing()
                            except Exception as e:
                                print(f"[WARNING] Batch {batch_start//MAX_SCANS_PER_BSCAN + 1} processing failed: {e}")
                    else:
                        # Generate all visualizations for smaller datasets
                        bscan = GPRBScanVisualizer(gpr.scan_log, gpr.gpr_detector)
                        bscan.generate_bscan_image()
                        bscan.generate_comprehensive_report()
                        
                        if BSCAN_SAVE_RAW_DATA:
                            bscan.export_data_for_external_processing()

                    print("[INFO] Generating flight log")
                    gpr.plot_full_flight_log()
                    gpr.write_log_to_csv()

                    print("[INFO] All GPR visualizations generated successfully")

                print("[INFO] Shutting down GPR detector...")
                gpr.shutdown()

            except Exception as e:
                print(f"[ERROR] Error during GPR cleanup: {e}")

        print("[INFO] Mission complete!")

# -*- coding: utf-8 -*-
#
#  FULL WEBOTS CONTROLLER WITH COMPLETE ROS1 INTEGRATION
#
from controller import Robot, Keyboard
from math import cos, sin
import time, sys, signal, json
import numpy as np
import matplotlib.pyplot as plt

# --- Your Custom Modules ---
from pid_controller import pid_velocity_fixed_height_controller
from object_detection import HumanDetector
from gpr_detection import GPRDetector
from gpr_bscan_plot import GPRBScanVisualizer
from gpr_detection import WebotsGPRInterface
from autonomous_navigator import AdvancedAutonomousNavigator

# ---------------- ROS IMPORTS ----------------
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool, String
from sensor_msgs.msg import Imu, NavSatFix, Image
# ---------------------------------------------

# CONSTANTS
FLYING_ATTITUDE = 1
SCAN_INTERVAL = 1.0
MIN_SCAN_DISTANCE = 0.2
MAX_SCANS_PER_BSCAN = 50
BSCAN_UPDATE_INTERVAL = 10

def signal_handler(sig, frame):
    print("\nGraceful shutdown...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ================================================================
# =====================    MAIN START     =========================
# ================================================================
if __name__ == "__main__":

    # Init Webots
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ----------- ROS INITIALIZATION -----------
    rospy.init_node("crazyflie_webots_full_node", anonymous=True)
    bridge = CvBridge()

    # ROS Publishers
    imu_pub = rospy.Publisher("/crazyflie/imu", Imu, queue_size=10)
    gps_pub = rospy.Publisher("/crazyflie/gps", NavSatFix, queue_size=10)
    altitude_pub = rospy.Publisher("/crazyflie/altitude", Float32, queue_size=10)

    range_front_pub = rospy.Publisher("/crazyflie/range/front", Float32, queue_size=10)
    range_left_pub  = rospy.Publisher("/crazyflie/range/left", Float32, queue_size=10)
    range_back_pub  = rospy.Publisher("/crazyflie/range/back", Float32, queue_size=10)
    range_right_pub = rospy.Publisher("/crazyflie/range/right", Float32, queue_size=10)

    camera_pub = rospy.Publisher("/crazyflie/camera/image_raw", Image, queue_size=10)

    human_detect_pub = rospy.Publisher("/crazyflie/human_detected", Bool, queue_size=10)
    human_count_pub = rospy.Publisher("/crazyflie/human_count", Float32, queue_size=10)

    gpr_detect_pub = rospy.Publisher("/crazyflie/gpr/detected", Bool, queue_size=10)
    gpr_conf_pub   = rospy.Publisher("/crazyflie/gpr/confidence", Float32, queue_size=10)

    autonomous_pub = rospy.Publisher("/crazyflie/autonomous/enabled", Bool, queue_size=10)
    mission_progress_pub = rospy.Publisher("/crazyflie/mission/progress", Float32, queue_size=10)

    # --------------------------------------------------------------

    # Initialize motors
    m1 = robot.getDevice("m1_motor"); m1.setPosition(float("inf")); m1.setVelocity(-1)
    m2 = robot.getDevice("m2_motor"); m2.setPosition(float("inf")); m2.setVelocity(+1)
    m3 = robot.getDevice("m3_motor"); m3.setPosition(float("inf")); m3.setVelocity(-1)
    m4 = robot.getDevice("m4_motor"); m4.setPosition(float("inf")); m4.setVelocity(+1)

    # Sensors
    imu = robot.getDevice("inertial_unit"); imu.enable(timestep)
    gps = robot.getDevice("gps"); gps.enable(timestep)
    gyro = robot.getDevice("gyro"); gyro.enable(timestep)
    camera = robot.getDevice("camera"); camera.enable(timestep)

    range_front = robot.getDevice("range_front"); range_front.enable(timestep)
    range_left  = robot.getDevice("range_left"); range_left.enable(timestep)
    range_back  = robot.getDevice("range_back"); range_back.enable(timestep)
    range_right = robot.getDevice("range_right"); range_right.enable(timestep)

    navigator = AdvancedAutonomousNavigator(gps, world_bounds=(-10, 30, -10, 30))
    detector = HumanDetector(camera)

    # GPR Detector
    try:
        gpr_detector = GPRDetector("gpr_models/template_human.in", "gpr_models/outputs", threshold=0.05)
        gpr = WebotsGPRInterface(gpr_detector, gps)
        gpr_available = True
    except:
        gpr = None
        gpr_available = False

    # PID Controller
    PID = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE

    # Keyboard
    kb = Keyboard(); kb.enable(timestep)

    # Variables
    last_x = last_y = 0
    last_time = 0
    first = True
    autonomous_mode = False
    gpr_scanning = True
    scan_counter = 0

    altitude_log = []
    time_log = []

    print("\nController + ROS started successfully.")

    # ============================================================
    # ====================== MAIN LOOP ===========================
    # ============================================================
    while robot.step(timestep) != -1 and not rospy.is_shutdown():

        current_time = robot.getTime()
        dt = current_time - last_time

        if first:
            last_x = gps.getValues()[0]
            last_y = gps.getValues()[1]
            last_time = current_time
            first = False
            continue

        # ------------------------------------------------------------
        # ============== SENSOR READINGS FROM WEBOTS =================
        # ------------------------------------------------------------

        roll, pitch, yaw = imu.getRollPitchYaw()
        yaw_rate = gyro.getValues()[2]

        x = gps.getValues()[0]
        y = gps.getValues()[1]
        z = gps.getValues()[2]

        vx = (x - last_x) / dt if dt > 0 else 0
        vy = (y - last_y) / dt if dt > 0 else 0

        altitude_log.append(z)
        time_log.append(current_time)

        # Body frame velocity
        vx_b = vx * cos(yaw) + vy * sin(yaw)
        vy_b = -vx * sin(yaw) + vy * cos(yaw)

        # Range sensors
        ranges = {
            "front": range_front.getValue() / 1000,
            "left":  range_left.getValue() / 1000,
            "back":  range_back.getValue() / 1000,
            "right": range_right.getValue() / 1000
        }

        # ------------------------------------------------------------
        # =================== PUBLISH TO ROS =========================
        # ------------------------------------------------------------

        # IMU
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.orientation.x = roll
        imu_msg.orientation.y = pitch
        imu_msg.orientation.z = yaw
        imu_msg.angular_velocity.z = yaw_rate
        imu_pub.publish(imu_msg)

        # GPS
        gps_msg = NavSatFix()
        gps_msg.header.stamp = rospy.Time.now()
        gps_msg.latitude = x
        gps_msg.longitude = y
        gps_msg.altitude = z
        gps_pub.publish(gps_msg)
        altitude_pub.publish(z)

        # Range sensors
        range_front_pub.publish(ranges["front"])
        range_left_pub.publish(ranges["left"])
        range_back_pub.publish(ranges["back"])
        range_right_pub.publish(ranges["right"])

        # Camera
        img = camera.getImage()
        if img:
            arr = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            ros_img = bridge.cv2_to_imgmsg(arr, encoding="rgba8")
            ros_img.header.stamp = rospy.Time.now()
            camera_pub.publish(ros_img)

        # Human Detection
        detector.detect_async()
        detections = detector.get_latest_detections()
        human_detect_pub.publish(len(detections) > 0)
        human_count_pub.publish(float(len(detections)))

        # GPR feedback
        if gpr_available:
            latest = gpr.get_latest_detection()
            if latest:
                gpr_detect_pub.publish(latest.has_detection)
                gpr_conf_pub.publish(latest.detection_confidence)

        # Autonomous mode
        autonomous_pub.publish(autonomous_mode)

        # Mission progress
        progress = navigator.get_coverage_progress()
        mission_progress_pub.publish(progress["progress_percentage"])

        # ------------------------------------------------------------
        # ============== KEYBOARD CONTROL ============================
        # ------------------------------------------------------------
        forward = sideways = yaw_des = height_diff = 0
        key = kb.getKey()

        while key > 0:
            if key == Keyboard.UP:    forward += 0.5
            if key == Keyboard.DOWN:  forward -= 0.5
            if key == Keyboard.RIGHT: sideways -= 0.5
            if key == Keyboard.LEFT:  sideways += 0.5
            if key == ord('W'):       height_diff = 0.1
            if key == ord('S'):       height_diff = -0.1
            if key == ord('Q'):       yaw_des = +1
            if key == ord('E'):       yaw_des = -1

            # Toggle autonomous
            if key == ord('A'):
                autonomous_mode = True
            if key == ord('D'):
                autonomous_mode = False

            key = kb.getKey()

        # Update autonomous navigation
        if autonomous_mode:
            nav_f, nav_s, nav_h = navigator.compute_navigation_control((x, y), current_time)
            forward += nav_f
            sideways += nav_s
            height_diff += nav_h

        height_desired += height_diff * dt

        # PID Controller
        motor_cmd = PID.pid(dt, forward, sideways, yaw_des,
                            height_desired, roll, pitch, yaw_rate,
                            z, vx_b, vy_b)

        # Apply to motors
        m1.setVelocity(-motor_cmd[0])
        m2.setVelocity(+motor_cmd[1])
        m3.setVelocity(-motor_cmd[2])
        m4.setVelocity(+motor_cmd[3])

        last_x, last_y, last_time = x, y, current_time


    print("\n[ROS] Node shutdown cleanly.")

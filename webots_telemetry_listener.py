#!/usr/bin/env python3
import rospy
import socket
import json

from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32, Int32

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

def fill_imu(msg, imu_rpy, gyro):
    msg.orientation.x = imu_rpy["roll"]
    msg.orientation.y = imu_rpy["pitch"]
    msg.orientation.z = imu_rpy["yaw"]
    msg.angular_velocity.x = gyro["x"]
    msg.angular_velocity.y = gyro["y"]
    msg.angular_velocity.z = gyro["z"]

if __name__ == "__main__":
    rospy.init_node("webots_telemetry_listener")

    gps_pub = rospy.Publisher("/webots/gps", Point, queue_size=10)
    imu_pub = rospy.Publisher("/webots/imu", Imu, queue_size=10)
    alt_pub = rospy.Publisher("/webots/altitude", Float32, queue_size=10)
    human_pub = rospy.Publisher("/webots/human_detections_count", Int32, queue_size=10)
    gpr_pub = rospy.Publisher("/webots/gpr_summary_json", Int32, queue_size=10)
    range_pub = rospy.Publisher("/webots/ranges", Point, queue_size=10)

    rospy.loginfo("webots_telemetry_listener running...")

    while not rospy.is_shutdown():
        data, addr = sock.recvfrom(65535)
        try:
            t = json.loads(data.decode())
        except:
            continue

        # GPS
        gps_msg = Point(t["gps"]["x"], t["gps"]["y"], t["gps"]["z"])
        gps_pub.publish(gps_msg)

        # IMU
        imu_msg = Imu()
        fill_imu(imu_msg, t["imu_rpy"], t["gyro"])
        imu_pub.publish(imu_msg)

        # Altitude
        alt_pub.publish(t["altitude"])

        # Human detection count
        human_pub.publish(t["human_detections_count"])

        # GPR summary
        if t["gpr_summary"] is not None:
            gpr_pub.publish(json.dumps(t["gpr_summary"]))

        # Ranges → encode in Point (x=front, y=left, z=right etc.)
        r = t["ranges"]
        range_pub.publish(Point(r["front"], r["left"], r["right"]))

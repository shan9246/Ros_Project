#!/usr/bin/env python3
import rospy
import socket
import json
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

WEBOTS_IP = "127.0.0.1"   # Windows side
WEBOTS_CMD_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

autonomous = False

def twist_callback(msg):
    """Map ROS Twist -> Webots forward/sideways/yaw commands."""
    cmd = {
        "forward": msg.linear.x,      # forward/backward
        "sideways": msg.linear.y,     # strafe
        "yaw": msg.angular.z,         # rotation
        "height_diff": msg.linear.z,  # up/down
        "autonomous": autonomous
    }

    sock.sendto(json.dumps(cmd).encode(), (WEBOTS_IP, WEBOTS_CMD_PORT))

def autonomous_callback(msg):
    global autonomous
    autonomous = msg.data

    cmd = {"autonomous": autonomous}
    sock.sendto(json.dumps(cmd).encode(), (WEBOTS_IP, WEBOTS_CMD_PORT))

def gpr_toggle_callback(msg):
    """Toggle GPR scanning: send flag."""
    cmd = {"gpr_toggle": True}
    sock.sendto(json.dumps(cmd).encode(), (WEBOTS_IP, WEBOTS_CMD_PORT))

def summary_callback(msg):
    """Request GPR summary from Webots."""
    cmd = {"request_summary": True}
    sock.sendto(json.dumps(cmd).encode(), (WEBOTS_IP, WEBOTS_CMD_PORT))

if __name__ == "__main__":
    rospy.init_node("webots_cmd_sender")

    rospy.Subscriber("/cmd_vel", Twist, twist_callback)
    rospy.Subscriber("/webots/autonomous_mode", Bool, autonomous_callback)
    rospy.Subscriber("/webots/gpr_toggle", Bool, gpr_toggle_callback)
    rospy.Subscriber("/webots/request_summary", Bool, summary_callback)

    rospy.loginfo("webots_cmd_sender running...")
    rospy.spin()

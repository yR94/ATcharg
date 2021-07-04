## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
from numba import jit, cuda
import pyrealsense2 as rs
import numpy as np
import cv2
import time

from robodk.robodk import Pose_2_TxyzRxyz, Mat, transl, rotz, rotx, roty
from robolink import *  # API to communicate with RoboDK for simulation and offline/online programming
from robodk import *  # Robotics toolbox for industrial robots

# Any interaction with RoboDK must be done through RDK:
from robolink.robolink import Robolink, ITEM_TYPE_ROBOT, RUNMODE_SIMULATE, ROBOTCOM_READY, RUNMODE_RUN_ROBOT

RDK = Robolink()
RDK.AddFile("New Station (1).rdk")

# Select a robot (popup is displayed if more than one robot is available)
robot = RDK.ItemUserPick('robot', ITEM_TYPE_ROBOT)
home = RDK.Item("home")
cam = RDK.Item("ABB_config_part")
if not robot.Valid():
    raise Exception('No robot selected or available')

RUN_ON_ROBOT = True

# Important: by default, the run mode is RUNMODE_SIMULATE
# If the program is generated offline manually the runmode will be RUNMODE_MAKE_ROBOTPROG,
# Therefore, we should not run the program on the robot
if RDK.RunMode() != RUNMODE_SIMULATE:
    RUN_ON_ROBOT = False

if RUN_ON_ROBOT:
    # Update connection parameters if required:
    # robot.setConnectionParams('192.168.2.35',30000,'/', 'anonymous','')

    # Connect to the robot using default IP
    success = robot.Connect()  # Try to connect once
    # success robot.ConnectSafe() # Try to connect multiple times
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        # Stop if the connection did not succeed
        print(status_msg)
        raise Exception("Failed to connect: " + status_msg)

    # This will set to run the API programs on the robot and the simulator (online programming)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)
#  RDK.setRunMode(RUNMODE_SIMULATE)
# Note: This is set automatically when we Connect() to the robot through the API

# else:
# This will run the API program on the simulator (offline programming)
# RDK.setRunMode(RUNMODE_SIMULATE)
# Note: This is the default setting if we do not execute robot.Connect()
# We should not set the RUNMODE_SIMULATE if we want to be able to generate the robot programm offline


# Get the current joint position of the robot
# (updates the position on the robot simulator)
robot.MoveJ(home)
joints_ref = robot.Joints()

# get the current position of the TCP with respect to the reference frame:
# (4x4 matrix representing position and orientation)

target_ref = robot.Pose()
pos_ref = target_ref.Pos()

#
# class targe:
#     def __init__(self, v):
#         self.x = v[0]
#         self.y = v[1]
#         self.z = v[2]
#         self.ox = v[3]
#         self.oy = v[4]
#         self.oz = v[5]
#


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


def yolo_fun(img, net, classes):
    Xcenter = 320
    Ycenter = 240

    # with open("coco.name", "r") as f:
    #   classes = f.read().splitlines()

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    # colors = np.array([0, 255, 255])

    center_x = Xcenter
    center_y = Ycenter
    alpha = 0

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    box_area = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.95:
                center_x = int(center_x * alpha + detection[0] * width * (1 - alpha))
                center_y = int(center_y * alpha + detection[1] * height * (1 - alpha))

                # center_x = int(detection[0] * width)
                # center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                box_area.append(w * h)
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)

    # a=indexes.flatten()

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = [250, 20, 20]  # colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
            box_area.append(h * w)

        max_box = max(box_area)
        ind_max_box = [i for i, j in enumerate(box_area) if j == max_box]
        x, y, w, h = boxes[ind_max_box[0]]
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        cv2.arrowedLine(img, (center_x, center_y), (Xcenter, Ycenter), (0, 255, 0), 2)
        cv2.circle(img, (center_x, center_y), 2, (0, 0, 222), -1)

        # print("x:   ",Xcenter-center_x,"Y:  ",Ycenter-center_y)
        # if(Xcenter-center_x<1 and Xcenter-center_x>-1 and Ycenter-center_y<1 and Ycenter-center_y>-1 ):
        #   print("good!!!!!")

        #  cv2.imshow('1', color_image[y:y + h,x:x + w])
        return [img, x, y, w, h]
    return [img, [], [], [], []]


def d_map2angle(xx, yy, zz):
    # zz = np.array(d_map)
    # xx,yy = np.meshgrid(np.arange(len(zz[0,:])),np.arange(len(zz[:,0])))## maybe xx and yy shulde start in th pix num (not in 0)

    zz = zz.flatten()
    xx = xx.flatten()
    yy = yy.flatten()

    b = np.where(zz > clipping_distance_in_meters)
    yy = np.delete(yy, b)
    xx = np.delete(xx, b)
    zz = np.delete(zz, b)

    Q = np.array([xx, yy, np.ones(len(zz))])
    abc = np.linalg.solve(np.dot(Q, np.transpose(Q)), np.dot(Q, zz.flatten()))
    theta_x = np.pi / 2 - np.arccos(abc[0] / np.sqrt(abc[0] ** 2 + 1))
    theta_y = np.pi / 2 - np.arccos(abc[1] / np.sqrt(abc[1] ** 2 + 1))

    return theta_x, theta_y, abc[2]


net = cv2.dnn.readNet('custom-station-detector_final.weights', 'custom-station-detector.cfg')

with open("obj.names", "r") as f:
    classes = f.read().splitlines()

try:
    targx = np.zeros(4)
    targy = np.zeros(4)
    targz = np.zeros(4)

    theta_x = np.zeros(4)
    theta_y = np.zeros(4)
    epsilon = 20
    while np.abs(epsilon) > 10:
        for i in [0, 1, 2, 3]:

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            # Stack both images horizontally
            ###############################################

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

            # Stack both images horizontally
            start = time.time()
            img, x, y, w, h = yolo_fun(color_image, net, classes)

            # images = np.hstack((color_image, depth_colormap))
            pc = rs.pointcloud()
            points = pc.calculate(aligned_depth_frame)
            pc.map_to(color_frame)
            vtx = np.asanyarray(points.get_vertices())

            x_matrix = np.array([x[0] for x in vtx]).reshape((480, 640))
            y_matrix = np.array([x[1] for x in vtx]).reshape((480, 640))
            z_matrix = np.array([x[2] for x in vtx]).reshape((480, 640))

            if not x == []:
                # s = int(0.5 * np.sqrt(w * h) / 2)
                xpw = int(x + w / 2 + w)
                xnw = int(x + w / 2 - w)
                ypw = int(y + h / 2 + h)
                ynw = int(y + h / 2 - h)
                if xnw < 0: xnw = 0
                if xpw > 640: xpw = 640
                if ynw < 0: ynw = 0
                if ypw > 480: ypw = 480
                if xnw > xpw or ynw > xpw: break
                s = - int(min(w, h) / 5)

                y_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0
                x_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0
                z_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0

                xx = x_matrix[(y - s):(y + h + s), (x - s):(x + w + s)]
                zz = z_matrix[(y - s):(y + h + s), (x - s):(x + w + s)]
                yy = y_matrix[(y - s):(y + h + s), (x - s):(x + w + s)]
                targx[i], targy[i], targz[i] = x_matrix[int(y + h / 2), int(x + w / 2)], y_matrix[
                    int(y + h / 2), int(x + w / 2)], z_matrix[int(y + h / 2), int(x + w / 2)] ### קורס באיטרצייה שנייה
                if not zz == []:
                    theta_x[i], theta_y[i], dist = d_map2angle(xx, yy, zz)
                    # print("OX:",theta_x,"OY",theta_y,dist)
                    # print("X:",targx,"Y:  ", targy,"Z: ", targz)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
        targx = np.mean(targx) * 1000
        targy = np.mean(targy) * 1000
        targz = np.mean(targz) * 1000-300  # set point dist at 200
        theta_x = np.mean(theta_x)
        theta_y = np.mean(theta_y)
        # It is important to provide the reference frame and the tool frames when generating programs offline
        # It is important to update the TCP on the robot mostly when using the driver
        epsilon =np.sqrt(targx**2+targy**2+targz**2)

        robot.setPoseFrame(robot.PoseFrame())
        robot.setPoseTool(robot.PoseTool())
        robot.setZoneData(10)  # Set the rounding parameter (Also known as: CNT, APO/C_DIS, ZoneData, Blending radius, cornering, ...)
        robot.setSpeed(50)  # Set linear speed in mm/s

        # Movement relative to the reference frame
        # Create a copy of the target
        target_ref = robot.Pose()
        target_i = Mat(target_ref)
        target_i = target_ref * transl(targy, -targx, targz) * rotx(-theta_x) * roty(-theta_y)  # (X=Ycam,Y=-Xcam,Z=Zcam)

        robot.MoveL(target_i)



finally:

    # Stop streaming
    pipeline.stop()
    print("fail")
    exit()

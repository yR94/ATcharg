## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

from robodk.robodk import Pose_2_TxyzRxyz, Mat,transl,rotz,rotx,roty
from robolink import *  # API to communicate with RoboDK for simulation and offline/online programming
from robodk import *  # Robotics toolbox for industrial robots

# Any interaction with RoboDK must be done through RDK:
from robolink.robolink import Robolink, ITEM_TYPE_ROBOT, RUNMODE_SIMULATE, ROBOTCOM_READY, RUNMODE_RUN_ROBOT

RDK = Robolink()
RDK.AddFile("old_orpt_v2.rdk")


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
#robot.MoveJ(home)
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

clipping_distance_in_meters = 1.7 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

def j1772detector(img):


    #img = cv2.imread("type_1_ev_inlet.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),1)
    blur_edges = cv2.Canny(blur,50,150,apertureSize = 3)

    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(blur_edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    # for i in range(4):
    #     for x1, y1, x2, y2 in lines[i]:
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    q = 15
    lower = np.array([0, 0, 0])
    upper = np.array([q, q, q])
    shapeMask = cv2.inRange(img, lower, upper)

    blur_shapeMask = cv2.GaussianBlur(shapeMask, (7, 7), 1)
    shapeMask_edges = cv2.Canny(blur_shapeMask, 50, 150)
    #shapeMask_edges = cv2.GaussianBlur(shapeMask_edges, (5, 5), 5)
    detected_circles = cv2.HoughCircles(shapeMask_edges,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30, minRadius=35, maxRadius=120)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        shapeMask_edges2=shapeMask_edges

        for pt in detected_circles[0, 0:1]:
            a, b, r = pt[0], pt[1], pt[2]


            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 1)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)




            #re
            #cv2.circle(shapeMask_edges2, (a, b), int(r*1.2), (0, 0, 0), -1)

    detected_circles = cv2.HoughCircles(shapeMask_edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5,maxRadius=80)



    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        # for pt in detected_circles[0, 0:1]:
        #     a, b, r = pt[0], pt[1], pt[2]
        #
        #     # Draw the circumference of the circle.
        #     cv2.circle(img, (a, b), r, (0, 255, 0), 1)
        #
        #     # Draw a small circle (of radius 1) to show the center.
        #     cv2.circle(img, (a, b), 1, (0, 0, 255), 2)
        #cv2.imshow('img4', blur_edges)
        cv2.imshow('img1', img)
        #cv2.imshow('img2', shapeMask)
        #cv2.imshow('img3', shapeMask_edges)
        cv2.waitKey(1)


    return [a,b]
def yolo_fun(img,net,classes):
    Xcenter=320
    Ycenter=240



    #with open("coco.name", "r") as f:
     #   classes = f.read().splitlines()


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    #colors = np.array([0, 255, 255])

    center_x = Xcenter
    center_y = Ycenter
    alpha = 0



    height, width, _ = img.shape


    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
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
                center_x = int(center_x * alpha + detection[0]*width * (1-alpha))
                center_y = int(center_y * alpha + detection[1]*height * (1-alpha))

                # center_x = int(detection[0] * width)
                # center_y = int(detection[1] * height)


                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                box_area.append(w*h)
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)

    #a=indexes.flatten()



    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = [250,20,20] #colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            box_area.append(h*w)

        max_box = max(box_area)
        ind_max_box = [i for i, j in enumerate(box_area) if j == max_box]
        x, y, w, h = boxes[ind_max_box[0]]
        center_x=int(x+w/2)
        center_y=int(y+h/2)
        cv2.arrowedLine(img, (center_x, center_y), (Xcenter, Ycenter), (0, 255, 0), 2)
        cv2.circle(img, (center_x, center_y), 2, (0, 0, 222), -1)

        #print("x:   ",Xcenter-center_x,"Y:  ",Ycenter-center_y)
        #if(Xcenter-center_x<1 and Xcenter-center_x>-1 and Ycenter-center_y<1 and Ycenter-center_y>-1 ):
         #   print("good!!!!!")

      #  cv2.imshow('1', color_image[y:y + h,x:x + w])
        return [img,x,y,w,h]
    return [img, [],[],[],[]]
def d_map2angle(xx,yy,zz):
   # zz = np.array(d_map)
    #xx,yy = np.meshgrid(np.arange(len(zz[0,:])),np.arange(len(zz[:,0])))## maybe xx and yy shulde start in th pix num (not in 0)



    zz=zz.flatten()
    xx=xx.flatten()
    yy=yy.flatten()

    b = np.where(zz > clipping_distance_in_meters)
    yy=np.delete(yy,b )
    xx=np.delete(xx, b)
    zz=np.delete(zz, b)

    Q=np.array([xx,yy,np.ones(len(zz))])
    abc=np.linalg.solve(np.dot(Q ,np.transpose(Q)),np.dot(Q,zz.flatten()))
    theta_x = np.pi/2 -np.arccos(abc[0]/np.sqrt(abc[0]**2+1))
    theta_y = np.pi/2 -np.arccos(abc[1] / np.sqrt(abc[1] ** 2 + 1))







    return theta_x,theta_y,abc[2]

net = cv2.dnn.readNet('custom-station-detector_final.weights', 'custom-station-detector.cfg')

with open("obj.names", "r") as f:
 classes = f.read().splitlines()

epsilon = 10
try:
      while epsilon > 5:
       a=4
       targx = np.zeros(a)
       targy = np.zeros(a)
       targz = np.zeros(a)

       theta_x = np.zeros(a)
       theta_y = np.zeros(a)

       for i in range(a):


            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
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

            img, x, y, w, h = yolo_fun(color_image, net, classes)



            #images = np.hstack((color_image, depth_colormap))
            pc = rs.pointcloud()
            points = pc.calculate(aligned_depth_frame)
            pc.map_to(color_frame)
            vtx = np.asanyarray(points.get_vertices())
            x_matrix = np.array([x[0] for x in vtx]).reshape((480, 640))
            y_matrix = np.array([x[1] for x in vtx]).reshape((480, 640))
            z_matrix = np.array([x[2] for x in vtx]).reshape((480, 640))

            if not x==[]:
                #s = int(0.5 * np.sqrt(w * h) / 2)
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

                xx = x_matrix[0:150, 0:150]
                zz = z_matrix[0:150, 0:150]
                yy = y_matrix[0:150, 0:150]
                targx[i],targy[i],targz[i] = x_matrix[int(y + h / 2), int(x + w / 2)], y_matrix[int(y + h / 2), int(x + w / 2)], z_matrix[ int(y + h / 2), int(x + w / 2)]
                if not zz == []:
                 theta_x[i], theta_y[i], dist = d_map2angle(xx, yy, zz)
                 print("OX:",theta_x,"OY",theta_y,dist)
                 print("X:",targx,"Y:  ", targy,"Z: ", targz)


            cv2.imshow('Image', img)
            cv2.waitKey(1)
       target_ref = robot.Pose()
       targx = int(np.mean(targx)*1000)#-52
       targy = int(np.mean(targy)*1000)
       targz = int(np.mean(targz)*1000-150 )#set point dist at 200)

       theta_x = np.mean(theta_x)
       theta_y = np.mean(theta_y)
       # It is important to provide the reference frame and the tool frames when generating programs offline
       # It is important to update the TCP on the robot mostly when using the driver
       robot.setPoseFrame(robot.PoseFrame())
       robot.setPoseTool(robot.PoseTool())
       robot.setZoneData(10)  # Set the rounding parameter (Also known as: CNT, APO/C_DIS, ZoneData, Blending radius, cornering, ...)
       robot.setSpeed(80)  # Set linear speed in mm/s


       norm = np.sqrt(targx**2+targy**2+targz**2)

       # Movement relative to the reference frame
       # Create a copy of the target
       targx_norm = 0.5*norm * targx / norm
       targy_norm = 0.5*norm * targy / norm
       targz_norm = 0.5*norm * targz / norm


       target_i = Mat(target_ref)
       epsilon_xy = 2
       epsilon_z = 300
       if targz > epsilon_z:
           target_i = target_ref * transl(targy_norm, -targx_norm, targz_norm)
       elif np.abs(targx) > epsilon_xy or np.abs(targy) > epsilon_xy:
            target_i = target_ref*transl(targy,-targx,0)#*rotx(-theta_x)*roty(-theta_y)#(X=Ycam,Y=-Xcam,Z=Zcam)
       else:
            target_i = target_ref*transl(0, 0, targz)#*rotx(-theta_x)*roty(-theta_y)#(X=Ycam,Y=-Xcam,Z=Zcam)

       # target_i = target_ref*transl(targy,-targx,0)#*rotx(-theta_x)*roty(-theta_y)#(X=Ycam,Y=-Xcam,Z=Zcam)
       # target_i = target_ref*transl(-100,0,0)#*rotx(-theta_x)*roty(-theta_y)#(X=Ycam,Y=-Xcam,Z=Zcam)
       epsilon = norm
       if abs(targx)<2 and abs(targy)<2 and targz < 1:
           epsilon = 1
           break

       robot.MoveL(target_i)



      for  i in range(1):
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

          img, x, y, w, h = yolo_fun(color_image, net, classes)
          xj,yj = j1772detector(img)

          pc = rs.pointcloud()
          points = pc.calculate(aligned_depth_frame)
          pc.map_to(color_frame)
          vtx = np.asanyarray(points.get_vertices())
          x_matrix = np.array([x[0] for x in vtx]).reshape((480, 640))
          y_matrix = np.array([x[1] for x in vtx]).reshape((480, 640))
          z_matrix = np.array([x[2] for x in vtx]).reshape((480, 640))

          jtargx, jtargy, jtargz = x_matrix[xj, yj], y_matrix[xj, yj], z_matrix[xj, yj]





          target_ref = robot.Pose()

          offz = 60
          offy = 35
          offx = -81

          robot.MoveL(target_ref * transl(jtargy + offx, 0, 0))
          target_ref = robot.Pose()
          robot.MoveL(target_ref * transl(0,-jtargx + offy, 0))
          target_ref = robot.Pose()
          robot.MoveL(target_ref * transl(0, 0, jtargz + offz))
          robot.setSpeed(80)
          target_ref = robot.Pose()
          time.sleep(3)
          robot.MoveL(target_ref * transl(0, 0,10))
          robot.setSpeed(10)
          target_ref = robot.Pose()
          robot.MoveL(target_ref * transl(0, 0, 10))









finally:

    # Stop streaming
    pipeline.stop()

print(exit)
exit()
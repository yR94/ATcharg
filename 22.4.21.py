## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import  time
import math
savedatax=[]
savedatay=[]
savedatadist=[]

class targe:
    def __init__(self, v):
        self.x = v[0]
        self.y = v[1]
        self.z = v[2]
        self.ox = v[3]
        self.oy = v[4]
        self.oz = v[5]





Xc=120
Yc=140



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

clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

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
    theta_x = 90 -np.rad2deg(np.arccos(abc[0]/np.sqrt(abc[0]**2+1)))
    theta_y = 90 - np.rad2deg(np.arccos(abc[1] / np.sqrt(abc[1] ** 2 + 1)))







    return theta_x,theta_y,abc[2]

net = cv2.dnn.readNet('custom-station-detector_final.weights', 'custom-station-detector.cfg')

with open("obj.names", "r") as f:
 classes = f.read().splitlines()


try:
    while True:


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
            s =  - int(min(w, h) / 10)



            y_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0
            x_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0
            z_matrix[np.where(z_matrix > clipping_distance_in_meters)] = 0


####### cheat: cheack the whit board surface
            xx = x_matrix[0:150, 0:150]
            zz = z_matrix[0:150, 0:150]
            yy = y_matrix[0:150, 0:150]




            targx,targy,targz = x_matrix[int(y + h / 2), int(x + w / 2)], y_matrix[int(y + h / 2), int(x + w / 2)], z_matrix[ int(y + h / 2), int(x + w / 2)]
            print(targx,targy,targz)

            if not zz == []:
             theta_x, theta_y, dist = d_map2angle(xx, yy, zz)
             print("OX:",theta_x,"OY",theta_y,dist)
            # print("X:",targx,"Y  ", targy,"Z: ", targz)

            from mpl_toolkits import mplot3d
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = plt.axes(projection='3d')


            ax.set_title('Surface plot')
            ax.scatter(x_matrix[int(y+h/2), int(x+w/2)], y_matrix[int(y+h/2), int(x+w/2)], z_matrix[int(y+h/2), int(x+w/2)], s=100,color = "r")
            ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')

            plt.show()



        cv2.imshow('Image', img)


        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            print(depth_image * depth_scale)
            cv2.destroyAllWindows()
            break

    plt.figure(1)
    plt.plot(range(len(savedatadist)),savedatax)
    plt.suptitle( np.mean(savedatax))
   # print("x - mean:   ",np.mean(savedatax),"    var:   ",np.var(savedatax))

    plt.figure(2)
    plt.plot(range(len(savedatadist)), savedatay)
    #print("y - mean:   ", np.mean(savedatay), "    var:   ", np.var(savedatay))


    plt.figure(3)
    plt.plot(range(len(savedatadist)), savedatadist)
    #print("dist - mean:   ", np.mean(savedatadist), "    var:   ", np.var(savedatadist))

    plt.figure(4)
    sp = np.fft.fft(np.array(savedatadist))
    freq = np.fft.fftfreq(np.arange(len(savedatadist)).shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)

    plt.show()

    with open('X_angle', 'w') as f:
        for item in savedatax:
            f.write("%s\n" % item)
    with open('Y_angle', 'w') as f:
        for item in savedatay:
            f.write("%s\n" % item)
    with open('../deepth/dist', 'w') as f:
        for item in savedatadist:
            f.write("%s\n" % item)


finally:

    # Stop streaming
    pipeline.stop()

import cv2
import numpy as np
import pyrealsense2 as rs
from time import  time
import matplotlib.pyplot  as plt

Xcenter = 320
Ycenter = 240


save_data_j=[]
save_data_y=[]
##### deepth cam
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
#print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

#####
def j1772_detector(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    blur_edges = cv2.Canny(blur, 50, 100, apertureSize=3)
    cv2.imshow("13", blur_edges)

    q = 15
    b_lower = np.array([0, 0, 0])
    b_upper = np.array([q, q, q])
    b_shapeMask = cv2.inRange(img, b_lower, b_upper)
    cv2.imshow("12", b_shapeMask)

    q = 50
    w_lower = np.array([q, q, q])
    w_upper = np.array([255,255,255])
    w_shapeMask = cv2.inRange(img, w_lower, w_upper)
    cv2.imshow("15", w_shapeMask)

    Wcircle = cv2.HoughCircles(blur_edges, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=20, minRadius=30, maxRadius=150)

    if Wcircle is not None:
        Wcircle = np.uint16(np.around(Wcircle))

        for pt in Wcircle[0, 0:1]:
            a1, b1, r1 = pt[0], pt[1], pt[2]

            a, b, r = pt[0], pt[1], pt[2]
            #cv2.circle(img, (a, b), 1, (0, 0, 255), 2)

            #cv2.circle(img, (a, b), r, (0, 0, 255), 1)

        return a,b

    cv2.imshow("11", img)
    return [], []

def d_map2angle(d_map):
    zz = np.array(d_map)
    xx,yy = np.meshgrid(np.arange(len(zz[0,:])),np.arange(len(zz[:,0])))## maybe xx and yy shulde start in th pix num (not in 0)
    Q=np.array([xx.flatten(),yy.flatten(),np.ones(len(zz[1,:])*len(zz[:,1]))])
    abc=np.linalg.solve(np.dot(Q ,np.transpose(Q)),np.dot(Q,zz.flatten()))
    theta_x = 90 -np.rad2deg(np.arccos(abc[0]/np.sqrt(abc[0]**2+1)))
    theta_y = 90 - np.rad2deg(np.arccos(abc[1] / np.sqrt(abc[1] ** 2 + 1)))
    #print("ox: " ,int(theta_x), '   ', "oy: ", int(theta_y), "   dist:  ", int(np.mean(check)))
    return theta_x,theta_y,abc[2]
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
    ##img = cv2.flip(img, 1)

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
            if confidence > 0.85:
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
       # print("x:   ",Xcenter-center_x,"Y:  ",Ycenter-center_y)
      # if(Xcenter-center_x<1 and Xcenter-center_x>-1 and Ycenter-center_y<1 and Ycenter-center_y>-1 ):
       #     print("good!!!!!")

        return [img,x,y,w,h,label]
    return [img, [],[],[],[],[]]






if __name__ == '__main__':
    net = cv2.dnn.readNet('custom-station-detector_final.weights', 'custom-station-detector.cfg')
    #classes = ["cover"]
    with open("obj.names", "r") as f:
     classes = f.read().splitlines()

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        last_time = time()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        img = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not img:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        img = np.asanyarray(img.get_data())
        clean_img = np.copy(img)



        img,x,y,w,h,label = yolo_fun(img,net,classes)

        if not x==[]:
         s=80
         if x < 0: x = 0
         if x > 640: x = 640
         if y < 0: y = 0
         if y > 480: y = 480



         check = np.array(depth_image)[320-s:320+s,240-s:240+s]#[(x-w):(x+w),(y-h):(y+h)]
         theta_x, theta_y, dist = d_map2angle(check)

         # a,b = j1772_detector(clean_img[(y):(y+h),(x):(x+w)])
         # if not(a == []):
         #
         #     cv2.circle(clean_img, (b+x,a+y), 1, (0, 0, 255), 2)
         #     save_data_j.append(b + x)
         #     save_data_y.append(a + y)





        cv2.imshow('Image2', clean_img)
        cv2.imshow('Image', img)



        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break



    cv2.destroyAllWindows()
    pipeline.stop()
with open('X', 'w') as f:
     for item in save_data_j:
                 f.write("%s\n" % item)
plt.figure(1)
plt.plot(range(len(save_data_j)), save_data_j)
plt.suptitle(np.mean(save_data_j))

plt.figure(2)
plt.plot(range(len(save_data_y)), save_data_y)
plt.suptitle(np.mean(save_data_y))
plt.show()
# with open('j1772_x', 'w') as f:
#      for item in save_data_j:
#             f.write("%s\n" % item)
# with open('j1772_y', 'w') as f:
#         for item in save_data_y:
#             f.write("%s\n" % item)

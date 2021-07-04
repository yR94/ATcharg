import cv2
import numpy as np
import pyrealsense2 as rs
from time import  time
from time import  sleep
import matplotlib.pyplot  as plt


Xcenter = 320
Ycenter = 240
counter = 0
alfa = 0.8


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
def j1772detector(img):
    # img = cv2.imread("type_1_ev_inlet.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    coler_avg = int(np.mean(blur) * 1.1)

    q = np.array(coler_avg)
    #
    upper = np.array([255, 255, 255])
    lower = np.array([q, q, q])
    shapeMask = cv2.inRange(img, lower, upper)

    blur_edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    mask_edges = cv2.Canny(shapeMask, 50, 150, apertureSize=3)

    detected_circles = cv2.HoughCircles(mask_edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5,
                                        maxRadius=130)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        shapeMask_edges2 = mask_edges

        for pt in detected_circles[0, 0:1]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 1)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)

    cv2.imshow('img', img)
    # cv2.imshow('img41', blur_edges)
    # cv2.imshow('img11', gray)
    # cv2.imshow('newMask', shapeMask)
    # cv2.imshow('mask_edges', mask_edges)

def d_map2angle(d_map):
    zz = np.array(d_map)
    xx,yy = np.meshgrid(np.arange(len(zz[0,:])),np.arange(len(zz[:,0])))## maybe xx and yy shulde start in th pix num (not in 0)

    # from mpl_toolkits import mplot3d
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    #
    # ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
    # ax.set_title('Surface plot')
    # plt.show()

    zz=zz.flatten()
    xx=xx.flatten()
    yy=yy.flatten()

    b = np.where(zz == 0)
    yy=np.delete(yy,b )
    xx=np.delete(xx, b)
    zz=np.delete(zz, b)

    Q=np.array([xx,yy,np.ones(len(zz))])
    abc=np.linalg.solve(np.dot(Q ,np.transpose(Q)),np.dot(Q,zz.flatten()))
    theta_x = 90 -np.rad2deg(np.arccos(abc[0]/np.sqrt(abc[0]**2+1)))
    theta_y = 90 - np.rad2deg(np.arccos(abc[1] / np.sqrt(abc[1] ** 2 + 1)))







    return theta_x,theta_y,abc[2]

def modes(label, x, y,theta_x, theta_y,dist ):
    global counter
    if label == []:
        if counter == -1:
            return
        print( 'mode 0')
        counter = counter + 1
        sleep(0.1)
        if counter > 4:
            print( 'mode 5')
          #  exit(1)
    elif label == 'cover':
        counter = -1
        print('cover')

        if dist<55 and dist > 25 and abs(x-Xcenter)<5  and abs(y-Ycenter)<5 and abs(theta_x) < 5 and  abs(theta_y) < 5:
            print('mode 2')
        else:
            print('mode 1')
            print('x  ', x - Xcenter, "y  ", y - Ycenter, 'ox  ', theta_x, "oy  ", theta_y)






    elif label == 'j1772':
        counter = -1
        print('j1772')
        if dist<300 and dist > 15 and abs(x-Xcenter)<150  and abs(y-Ycenter)<150 :
            print('mode 4')






        else:
            print('mode 3')
            print('x  ', x-Xcenter, "y  ", y-Ycenter, 'ox  ', theta_x, "oy  ", theta_y)
    return



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
        theta_x, theta_y, dist = [],[],[]

        if not label==[]:

            # xpw = int(x + w/2+w)
            # xnw = int(x + w/2-w)
            # ypw = int(y + w/2+h)
            # ynw =int( y + w/2-h)
            # if xnw < 0: xnw = 0
            # if xpw > 640: xpw = 640
            # if ynw < 0: ynw = 0
            # if ypw > 480: ypw = 480
            # if xnw>xpw or ynw>xpw: break

            check = np.array(depth_image)[ y:y+h,x:x+w]
            cv2.imshow('1', color_image[ y:y+h,x:x+w])



           # check = np.array(depth_image) #[Xcenter-w:Xcenter+w,Ycenter-h:Ycenter+h]  #[(x-w):(x+w),(y-h):(y+h)]
            theta_x, theta_y, dist = d_map2angle(check)

         # a,b = j1772_detector(clean_img[(y):(y+h),(x):(x+w)])
         # if not(a == []):
         #
         #     cv2.circle(clean_img, (b+x,a+y), 1, (0, 0, 255), 2)
         #     save_data_j.append(b + x)
         #     save_data_y.append(a + y)

        modes(label, x+w, y+h, theta_x, theta_y, dist)





      #  cv2.imshow('Image2', clean_img)
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

import cv2
import numpy as np
import matplotlib.pyplot as plt
savedatax=[]
savedatay=[]

cap=cv2.VideoCapture(1)
cap.set(10,160)
savedata = np.array([])

aa,bb,rr = 0,0,0
aa1,bb1,rr1 = 0,0,0
aa2,bb2,rr2 = 0,0,0
alfa = 0.8
astak =[]


while True:

    # Load an image
    success, img = cap.read()

    #img = cv2.imread("type_1_ev_inlet.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),1)
    blur_edges = cv2.Canny(blur,50,150,apertureSize = 3)
    cv2.imshow('blur_edges', blur_edges)




    q = 20
    lower = np.array([0, 0, 0])
    upper = np.array([q, q, q])
    shapeMask = cv2.inRange(img, lower, upper)


    blur_shapeMask = cv2.GaussianBlur(shapeMask, (7, 7), 1)
    shapeMask_edges = cv2.Canny(blur_shapeMask, 50, 150)
    #shapeMask_edges = cv2.GaussianBlur(shapeMask_edges, (5, 5), 5)
    detected_circles = cv2.HoughCircles(blur_edges,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30, minRadius=35, maxRadius=120)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        shapeMask_edges2=shapeMask_edges

        for pt in detected_circles[0, 0:1]:
            a, b, r = pt[0], pt[1], pt[2]

            aa = int(aa*alfa+a*(1-alfa))
            bb = int(bb * alfa + b * (1 - alfa))
            rr = int(rr * alfa + r * (1 - alfa))
            a,b,r =aa,bb,rr

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 1)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)
            x1,x2,y1,y2 = [int(b-r*1.1),int(b+r*1.1),int(a-r*1.1),int(a+r*1.1)]
            newframe = gray[x1:x2,y1:y2]
            newframe2 = gray[int(b - r +5):int(b + r-5), int(a - r+5):int(a + r -5)]

            newblur = cv2.GaussianBlur(newframe, (5,5), 1)
            newcanny = cv2.Canny(newframe, 50, 150, apertureSize=3)
            backtorgb = cv2.cvtColor(newframe, cv2.COLOR_GRAY2RGB)
            q = 10
            lower = np.array([q, q, q])
            upper = np.array([255, 255, 255])
            newMask = cv2.inRange(backtorgb, lower, upper)
            newcanny2 = cv2.Canny(newMask, 50, 150, apertureSize=3)




            Wcircle = cv2.HoughCircles (newcanny, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=30, minRadius=50, maxRadius=180)
            Wcircle2 = cv2.HoughCircles(newcanny, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=30, minRadius=5,  maxRadius=80)


            if Wcircle is not None:
                Wcircle = np.uint16(np.around(Wcircle))

                for ptt in Wcircle[0, 0:1]:
                    a1, b1, r1 = ptt[0], ptt[1], ptt[2]
                    aa1 = int(aa1 * alfa + a1 * (1 - alfa))
                    bb1 = int(bb1 * alfa + b1 * (1 - alfa))
                    rr1 = int(rr1 * alfa + r1 * (1 - alfa))
                    a1, b1, r1 = aa1, bb1, rr1

                    # Draw a small circle (of radius 1) to show the center.
                    cv2.circle(backtorgb, (a1, b1), 1, (0, 0, 255), 2)
                    cv2.circle(backtorgb, (a1, b1), r1, (0, 0, 255), 2)
                    cv2.circle(img, (a1+y1, b1+x1), 1, (0, 0, 255), 2)
                    cv2.circle(img, (a1+y1, b1+x1), r1, (0, 0, 255), 2)




            cv2.imshow('1', newcanny)
            # cv2.imshow('3', newMask)

            if Wcircle2 is not None:
                Wcircle2 = np.uint16(np.around(Wcircle2))

                for ptt2 in Wcircle2[0, 0:1]:
                    a2, b2, r2 = ptt2[0], ptt2[1], ptt2[2]

                    aa2 = int(aa2 * alfa + a2 * (1 - alfa))
                    bb2 = int(bb2 * alfa + b2 * (1 - alfa))
                    rr2 = int(rr2 * alfa + r2 * (1 - alfa))
                    a2, b2, r2 = aa2, bb2, rr2

                    # Draw a small circle (of radius 1) to show the center.
                    cv2.circle(backtorgb, (a2, b2), 1, (255, 0, 255), 2)
                    cv2.circle(backtorgb, (a2, b2), r2, (255, 0, 255), 2)
                    cv2.circle(img, (a2 + y1, b2 + x1), 1, (0, 255, 255), 2)
                    cv2.circle(img, (a2 + y1, b2 + x1), r2, (0, 255, 255), 2)
                    savedatay.append(a2 + y1)
                    savedatax.append(b2 + x1)


            cv2.imshow('1', newcanny)



            cv2.imshow('2', shapeMask)








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



    cv2.imshow('img1', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:

        cv2.destroyAllWindows()
        break


# with open('circles_x_on_0.10', 'w') as f:
#      for item in savedatax:
#             f.write("%s\n" % item)
# with open('circles_y_on_0.10', 'w') as f:
#         for item in savedatay:
#             f.write("%s\n" % item)
plt.figure(1)
plt.plot(range(len(savedatax)),savedatax)
plt.figure(1)
plt.plot(range(len(savedatay)),savedatay)
plt.show()
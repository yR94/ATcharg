import cv2
import numpy as np

cap=cv2.VideoCapture(1)

center_position = []
while True:

    # Load an image
    success, img = cap.read()
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



    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])

    shapeMask = cv2.inRange(img, lower, upper)

    blur_shapeMask = cv2.GaussianBlur(shapeMask, (5, 5), 5)
    shapeMask_edges = cv2.Canny(blur_shapeMask, 50, 150)
    detected_circles = cv2.HoughCircles(shapeMask_edges,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30, minRadius=5, maxRadius=50)

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
            cv2.circle(shapeMask_edges2, (a, b), int(r*1.2), (0, 0, 0), -1)

    detected_circles = cv2.HoughCircles(shapeMask_edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5,maxRadius=80)



    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 1)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 2)


    cv2.imshow('img4',blur_edges)
    cv2.imshow('img1', img)
    cv2.imshow('img2', shapeMask)
    cv2.imshow('img3', shapeMask_edges)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyWindow()
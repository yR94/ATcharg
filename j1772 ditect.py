import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:

    # Load an image
    _, img = cap.read()
    im = img
    im_circles = img
    im_extreme = img

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blue = im[..., 0]
    equ = cv2.equalizeHist(im_gray)
    _, threshold = cv2.threshold(equ, 20, 255, cv2.THRESH_BINARY)

    # ----------------------------------------------------
    MORPH = 3
    CANNY = 250
    kernel = np.ones((3, 3), np.uint8)
    im_gray_blur = cv2.GaussianBlur(im_gray, (3, 3), 1)
    edges = cv2.Canny(im_gray_blur, 50, 100)
    edges = cv2.dilate(edges, kernel, iterations=1)
    im_gray_negativ = 255 - im_gray_blur
    edges_negativ = 255 - edges
    _, negativ_threshold = cv2.threshold(im_gray_negativ, 210, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        Area = cv2.contourArea(cnt)
        if (Area > 800):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            objCor = len(approx)
            if objCor > 7:
                cv2.drawContours(im, [cnt], 0, (0, 255, 0), 1)

    print(len(contours))
    cv2.imshow("im", im)

    # ----------------------find circles------------------------------

    # Load an image
    image_gray = im_gray
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(edges, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(edges,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=40, maxRadius=80)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(im_circles, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(im_circles, (a, b), 1, (0, 0, 255), 3)
    cv2.imshow("Detected Circle", im_circles)

    # ----------------------find extreme points------------------------------
    # import the necessary packages
    import imutils

    # load the image, convert it to grayscale, and blur it slightly
    negativ_blue = 255 - blue
    blue_blur = cv2.GaussianBlur(blue, (5, 5), 0)
    image1 = im
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise

    edges = cv2.Canny(blue_blur, 50, 100)

    thresh = cv2.threshold(negativ_blue, 80, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(im_extreme, [c], -1, (0, 255, 255), 2)
    cv2.circle(im_extreme, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(im_extreme, extRight, 8, (0, 255, 0), -1)
    cv2.circle(im_extreme, extTop, 8, (255, 0, 0), -1)
    cv2.circle(im_extreme, extBot, 8, (255, 255, 0), -1)
    # show the output image
    cv2.imshow("edges", edges)
    cv2.imshow("thresh", thresh)
    cv2.imshow("Image", im_extreme)


    cv2.imshow("img",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
# img = cv2.imread('img3.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray,5)
#
#
# # define range of blue color in HSV
# lower_blue = np.array([1, 1, 1])
# upper_blue = np.array([120, 120, 120])
# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image

# circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20, param1=60,param2=30,minRadius=0,maxRadius=50)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
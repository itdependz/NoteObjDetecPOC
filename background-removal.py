import cv2
import numpy as np

myimage = cv2.imread('licensed-image.jpeg')

# First Convert to Grayscale
myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)

ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)

ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)

ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)

foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground

# Convert black and white back into 3 channel greyscale
background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

# Combine the background and foreground to obtain our final image
finalimage = background+foreground

cv2.imshow('Original Image', myimage)
cv2.waitKey(0)
cv2.imshow('Greyscale Image', myimage_grey)
cv2.waitKey(0)
cv2.imshow('Baseline Image', baseline)
cv2.waitKey(0)
cv2.imshow('Background', background)
cv2.waitKey(0)
cv2.imshow('Foreground', foreground)
cv2.waitKey(0)
cv2.imshow('Final Image', finalimage)
cv2.waitKey(0)
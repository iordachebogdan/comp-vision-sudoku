"""Script for showing the template cube and printing the coordinates of pixels
based on mouse click events
"""
import cv2 as cv

TEMPLATE_PATH = "./template.jpg"


def print_coords(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)


template = cv.imread(TEMPLATE_PATH, cv.IMREAD_GRAYSCALE)
cv.imshow("Template", template)
cv.setMouseCallback("Template", print_coords)
cv.waitKey(0)
cv.destroyAllWindows()

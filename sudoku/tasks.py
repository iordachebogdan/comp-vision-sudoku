import cv2 as cv
import matplotlib.pyplot as plt

from sudoku.image_processing import (
    preprocess_binary_image,
    find_largest_polygon,
    dilate_image,
    get_corners,
    warp_image,
    classify_boxes,
    CompletedBoxClassifier,
)
from sudoku.generic_utils import show_image_pyplot


def make_prediction_task1(img, debug=False):
    """Detect which cells are completed in the sudoku board contained in the given image

    Parameters:
        img (ndarray): original image
        debug (bool): debugging flag

    Returns:
        parsed_board (str): the parsing of the sudoku board based on the task 1
            description
    """
    if debug:
        show_image_pyplot(img)

    img = img.copy()
    # preprocess image in order to detect edges
    preprocessed_img = preprocess_binary_image(img)
    if debug:
        show_image_pyplot(preprocessed_img)
    plt.clf()

    # dilate binary image to make it easier to detect the sudoku square
    dilated_img = dilate_image(preprocessed_img)
    if debug:
        show_image_pyplot(dilated_img)
    plt.clf()

    # assume that the largest found polygon is the sudoku square
    sudoku_square = find_largest_polygon(dilated_img)
    found_contour_img = cv.drawContours(img, [sudoku_square], 0, (0, 255, 0), 3)
    tr, tl, bl, br = get_corners(sudoku_square)
    for corner in [tr, tl, bl, br]:
        found_contour_img = cv.circle(
            found_contour_img, (corner[0], corner[1]), 10, (255, 0, 0), thickness=-1
        )
    if debug:
        show_image_pyplot(found_contour_img)
    plt.clf()

    # warp perspective for the found sudoku square
    warped = warp_image(img, corners=(tr, tl, bl, br))
    # convert it to graysacle
    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    if debug:
        show_image_pyplot(warped_gray)
    plt.clf()

    # classify the cells based on the variance of the pixels (completed cells
    # contain both very dark and very bright pixels)
    classifier = CompletedBoxClassifier(thresh=600, mode="var")
    predict_str = classify_boxes(warped_gray, classifier, debug=debug)
    plt.clf()

    return predict_str

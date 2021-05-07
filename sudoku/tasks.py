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
    remove_light_grays,
    classify_borders,
    fill_regions,
    find_colored_zones,
    has_color,
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


def combine_predictions(pred_digits, pred_zones):
    """For task 2 combine the prediction for digits with the one for regions"""
    res = ""
    for fst, snd in zip(pred_zones.split("\n"), pred_digits.split("\n")):
        for x, y in zip(fst, snd):
            res += x + y
        res += "\n"
    return res[:-1]


def make_prediction_task2_gray(warped, debug=False):
    """Make predictions for task 2 when the detected jigsaw puzzle has white cells.

    Parameters:
        warped (ndarray): the warped jigsaw sudoku square
        debug (bool): debugging flag

    Returns:
        parsed_board (str): the parsing of the jigsaw sudoku board based on the
            task 2 description
    """

    # remove light grays from the image in order to remain with the thicker borders
    grays_removed = remove_light_grays(warped)
    grays_removed = cv.medianBlur(grays_removed, 5)
    if debug:
        show_image_pyplot(grays_removed)

    # compute the binary representation of the board
    binary_warped = preprocess_binary_image(grays_removed, kernel_size=(3, 3))
    if debug:
        show_image_pyplot(binary_warped)

    # dilate the binary image in order to make the remaining borders even thicker
    dilated_image = dilate_image(binary_warped, kernel_size=7)
    if debug:
        show_image_pyplot(dilated_image)

    # classify which cells contain digits
    digit_classifier = CompletedBoxClassifier(thresh=600, mode="var")
    gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    predicted_digits = classify_boxes(gray_warped, digit_classifier, debug=debug)

    # determin the borders on the side of each cell, here we look at the binary image
    # and determine the existence of a border based on the density of white pixels
    border_classifier = CompletedBoxClassifier(thresh=100, mode="mean")
    borders = classify_borders(
        dilated_image, border_classifier, debug=debug, margin_pct=0.10
    )
    predicted_regions = fill_regions(borders)

    final_predictions = combine_predictions(predicted_digits, predicted_regions)
    if debug:
        print(final_predictions)

    return final_predictions


def make_prediction_task2_color(warped, debug=False):
    """Make predictions for task 2 when the detected jigsaw puzzle has colored cells.

    Parameters:
        warped (ndarray): the warped jigsaw sudoku square
        debug (bool): debugging flag

    Returns:
        parsed_board (str): the parsing of the jigsaw sudoku board based on the
            task 2 description
    """

    binary_warped = preprocess_binary_image(warped, kernel_size=(5, 5))
    binary_warped = dilate_image(binary_warped, kernel_size=4)
    if debug:
        show_image_pyplot(binary_warped)

    # classify which cells contain digits
    digit_classifier = CompletedBoxClassifier(thresh=600, mode="var")
    gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    predicted_digits = classify_boxes(gray_warped, digit_classifier, debug=debug)

    # predict the regions using the color of the cells
    predicted_regions = find_colored_zones(warped, binary_warped, debug=debug)

    final_predictions = combine_predictions(predicted_digits, predicted_regions)
    if debug:
        print(final_predictions)

    return final_predictions


def make_prediction_task2(img, debug=False):
    """Detect which cells are completed in the jigsaw sudoku board contained in
    the given image and also label the detected jigsaw regions.

    Parameters:
        img (ndarray): original image
        debug (bool): debugging flag

    Returns:
        parsed_board (str): the parsing of the sudoku board based on the task 1
            description
    """
    if debug:
        show_image_pyplot(img)

    # preprocess image in order to detect edges
    preprocessed_img = preprocess_binary_image(img)
    if debug:
        show_image_pyplot(preprocessed_img)

    # assume that the largest found polygon is the sudoku square
    sudoku_square = find_largest_polygon(preprocessed_img)
    found_contour_img = cv.drawContours(img.copy(), [sudoku_square], 0, (0, 255, 0), 3)
    tr, tl, bl, br = get_corners(sudoku_square)
    for corner in [tr, tl, bl, br]:
        found_contour_img = cv.circle(
            found_contour_img, (corner[0], corner[1]), 10, (255, 0, 0), thickness=-1
        )
    if debug:
        show_image_pyplot(found_contour_img)

    # warp perspective for the found sudoku square
    warped = warp_image(img, corners=(tr, tl, bl, br))
    if debug:
        show_image_pyplot(warped)

    # here we check in which of the 2 cases the current image is (colored jigsaw
    # or black and white jigsaw)
    if has_color(warped):
        return make_prediction_task2_color(warped, debug=debug)
    else:
        return make_prediction_task2_gray(warped, debug=debug)

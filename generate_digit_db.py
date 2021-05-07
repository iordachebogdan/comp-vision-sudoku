""" Script for loading one of the training images for task 3 and extract a database
with one example from each of the 9 possible digits. These examples are processed
into binary images and then dilated
"""
import cv2 as cv

from sudoku.image_processing import (
    preprocess_binary_image,
    find_largest_polygon,
    get_corners,
    dilate_image,
    warp_image,
    get_digit_cell,
)
from sudoku.generic_utils import show_image_pyplot

IMAGE_PATH = "./train/cube/1.jpg"


def main():
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    # preprocess image and extract one of the sudoku squares
    preprocessed_img = preprocess_binary_image(img)
    sudoku_square = find_largest_polygon(preprocessed_img)
    tr, tl, bl, br = get_corners(sudoku_square)
    warped = warp_image(img, corners=(tr, tl, bl, br), side_len=180)

    # binarize the extracted sudoku square and apply dilation
    applied_blur = cv.GaussianBlur(warped, (3, 3), 0)
    thresholded = cv.adaptiveThreshold(
        applied_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2
    )
    inverted = cv.bitwise_not(thresholded)
    warped_binary = dilate_image(inverted, 2)
    show_image_pyplot(warped_binary)

    # example for each digit from 1 to 9
    cells = [
        get_digit_cell(warped_binary, 0, 6),
        get_digit_cell(warped_binary, 0, 5),
        get_digit_cell(warped_binary, 4, 0),
        get_digit_cell(warped_binary, 0, 3),
        get_digit_cell(warped_binary, 4, 1),
        get_digit_cell(warped_binary, 0, 4),
        get_digit_cell(warped_binary, 0, 8),
        get_digit_cell(warped_binary, 1, 8),
        get_digit_cell(warped_binary, 0, 7),
    ]

    for i, cell in enumerate(cells):
        cv.imwrite(f"./digit_db/{i+1}.bmp", cell)


if __name__ == "__main__":
    main()

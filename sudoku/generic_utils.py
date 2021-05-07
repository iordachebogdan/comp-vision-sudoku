import os
import re

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection

from sudoku.image_processing import resize_image


def show_image_opencv(image, window_name="window", timeout=0):
    """Show image inside an OpenCV window"""
    cv.imshow(window_name, np.uint8(image))
    cv.waitKey(timeout)
    cv.destroyAllWindows()


def show_image_pyplot(image):
    """Show image using pyplot"""
    if len(image.shape) == 3:
        plt.imshow(image[:, :, [2, 1, 0]])
    else:
        plt.imshow(image, cmap="gray")
    plt.show()


def load_images(input_dir, resize=None, grayscale=False):
    """Load images from input directory. It assumes that the filenames of the
    images are {number}.jpg

    Parameters:
        input_dir (str): path to input directory
        resize (int, optional): resize images to specified width
        grayscale (bool): load images as grayscale

    Returns:
        (images, filenames) (list, list): loaded images and their corresponding
            filenames
    """
    filenames = set(
        [
            re.split(r"\.|_", filename)[0]
            for filename in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, filename))
        ]
    )
    # filter filenames that are numbers
    filenames = [filename for filename in filenames if filename.isdigit()]
    filenames = sorted(filenames, key=lambda x: int(x))

    all_images = [
        cv.imread(
            os.path.join(input_dir, f"{filename}.jpg"),
            cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR,
        )
        for filename in filenames
    ]
    if resize is not None:
        all_images = [resize_image(image, resize) for image in all_images]
    print(f"Loaded {len(all_images)} images")

    return all_images, filenames


def train_test_split(all_images, all_gt, test_size=0.2):
    """Split images and ground truths into train and test"""
    train_images, test_images, train_gt, test_gt = model_selection.train_test_split(
        all_images, all_gt, test_size=test_size, random_state=42
    )
    return train_images, test_images, train_gt, test_gt

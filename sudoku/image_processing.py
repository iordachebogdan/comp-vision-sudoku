import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance


def resize_image(image, d_width, inter=cv.INTER_LANCZOS4):
    """Resize image based on target width while maintaining the aspect ratio"""
    height, width = image.shape[:2]
    d_height = int(height * d_width / width)
    return cv.resize(image, dsize=(d_width, d_height), interpolation=inter)


def remove_light_grays(img):
    """Removes light gray pixels from the image by replacing them with white pixels.
    The pixels are identified by low saturation (lack of color) and high brightness
    (since we do not want to remove black pixels).
    """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, (0, 0, 100), (255, 20, 255))
    res = img.copy()
    res[np.where(mask)] = (255, 255, 255)
    return res


def preprocess_binary_image(image, kernel_size=(7, 7)):
    """Preprocess given image and generate a binary one, emphasizing the edges.
    This is done by converting the image to grayscale, applying a Gaussian blur,
    followed by an adaptive threshold. The resulted binary image is then inverted,
    such that the edges become white and a median filter is applied in order to
    remove "salt and pepper"-like noise.

    Parameters:
        image (ndarray): image to preprocess
        kernel_size (tuple): kernel dimensions used for gaussian blur and adaptive
            threshold

    Returns:
        binary_image (ndarray): the binary image resulted
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv.GaussianBlur(gray, kernel_size, 0)
    thresholded = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel_size[0], 2
    )
    inverted = cv.bitwise_not(thresholded)
    median_blur = cv.medianBlur(inverted, 5)
    return median_blur


def dilate_image(binary_img, kernel_size=3, iterations=1):
    """Dilate given image using a square kernel of the specified dimension

    Parameters:
        binary_img (ndarray): binary image we wish to dilate
        kernel_size (int): dimension of dilation square kernel
        iterations (int): number of dilation iterations

    Returns:
        dilated_img (ndarray): the dilated binary image
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv.dilate(binary_img, kernel, iterations=iterations)


def find_largest_polygon(binary_img):
    """Returns the largest polygon (contour), based on area,
    found in the given binary image
    """

    # we use RETR_EXTERNAL since we do not need to consider polyongs included in
    # larger ones
    polygons, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polygon = max(polygons, key=lambda p: cv.contourArea(p))
    return polygon


def get_corners(polygon):
    """Assuming the given polygon approximates a rectangle return its four corners

    Parameters:
        polygon (cv2 polygon)

    Returns:
        (top_right, top_left, bottom_left, bottom_right): corners
    """

    def get_max_vertex(cmp):
        return max([(vertex[0][0], vertex[0][1]) for vertex in polygon], key=cmp)

    tr = get_max_vertex(lambda p: p[0] - p[1])
    tl = get_max_vertex(lambda p: -p[0] - p[1])
    bl = get_max_vertex(lambda p: -p[0] + p[1])
    br = get_max_vertex(lambda p: p[0] + p[1])

    return (tr, tl, bl, br)


def compute_longest_side(tr, tl, bl, br):
    """Returns length of longest side of a quad, given its four vertices"""
    return max(
        distance.euclidean(tr, tl),
        distance.euclidean(bl, tl),
        distance.euclidean(br, bl),
        distance.euclidean(br, tr),
    )


def warp_image(img, corners, side_len=None):
    """Given the corners of a rectangle inside an image, crop it from the image,
    and warp the perspective to obtain an axis aligned square.

    Parameters:
        img (ndarray): original image
        corners (tuple): top right, top left, bottom left, bottom right corners of
            the rectangle
        side_len (int, optional): resulted side length of the warped square, by default
            use the length of the longest side of the detected rectangle

    Returns:
        warped_square (ndarray): the warped square
    """
    tr, tl, bl, br = corners
    edge_len = (
        side_len if side_len is not None else int(compute_longest_side(tr, tl, bl, br))
    )

    transform_matrix = cv.getPerspectiveTransform(
        np.array([tl, tr, br, bl], dtype=np.float32),
        np.array(
            [
                [0, 0],
                [edge_len - 1, 0],
                [edge_len - 1, edge_len - 1],
                [0, edge_len - 1],
            ],
            dtype=np.float32,
        ),
    )

    return cv.warpPerspective(img, transform_matrix, (edge_len, edge_len))


class CompletedBoxClassifier:
    """Classifier for images based on the mean or variance computed for their pixels"""

    def __init__(self, thresh, mode="var"):
        """
        Parameters:
            thresh (number): threshold above which the image is considered from the
                positive class
            mode (str): "var" or "mean", what metric to use
        """
        self.thresh = thresh
        self.mode = mode

    def __call__(self, box):
        if self.mode == "var":
            return np.var(box) > self.thresh
        elif self.mode == "mean":
            return np.mean(box) > self.thresh


def classify_boxes(img, classifier, debug=False, margin_pct=0.20):
    """Given the warped image of the sudoku square, split it equally into 81 cells
    and detect which ones contain digits.

    Parameters:
        img (ndarray): warped image of a sudoku square
        classifier (CompletedBoxClassifier): classifier for completed cells
        debug (bool): debugging flag
        margin_pct (float): for each cell crop the margins based on this percentage,
            in order to ignore the borders of the sudoku grid

    Returns:
        parsed_board (str): the parsing of the sudoku boards showing which cells
            are completed
    """

    # length of the side of a cell
    box_len = int(img.shape[0] / 9)

    return_string = ""

    fig = plt.figure(figsize=(9, 9))
    fig.tight_layout()
    for i in range(9):
        for j in range(9):
            # extract cell
            box = img[
                i * box_len : (i + 1) * box_len - 1, j * box_len : (j + 1) * box_len - 1
            ].copy()
            margin = int(box_len * margin_pct)
            # trim the margins
            box = box[margin:-margin, margin:-margin]

            predicted = classifier(box)

            fig.add_subplot(9, 9, i * 9 + j + 1)
            if predicted:
                plt.imshow(box)
                return_string += "x"
            else:
                plt.imshow(box, cmap="gray")
                return_string += "o"
            plt.axis("off")
        return_string += "\n"

    if debug:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)

    return return_string[:-1]

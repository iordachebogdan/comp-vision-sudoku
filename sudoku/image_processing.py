import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from sklearn.cluster import KMeans


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
        parsed_board (str): the parsing of the sudoku board showing which cells
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


def has_color(img):
    """Check if given image contains a colored jigsaw sudoku. This is decided
    based on the mean saturation of the image. A higher saturation means more
    vibrant colors"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1]) > 15


class CellBorders:
    """Store for a given sudoku cell if there are borders of a jigsaw region on
    either of its four sides"""

    def __init__(self, up=False, right=False, down=False, left=False):
        self.up = up
        self.right = right
        self.down = down
        self.left = left

    def __str__(self):
        res = ""
        if self.up:
            res += "U"
        if self.right:
            res += "R"
        if self.down:
            res += "D"
        if self.left:
            res += "L"
        if not res:
            res = "none"
        return res

    def __repr__(self):
        return str(self)


def classify_borders(binary_img, classifier, debug=False, margin_pct=0.10):
    """Given the warped binary image of the jigsaw sudoku square, split it equally
    into 81 cells and detect which ones determine the borders of the jigsaw regions.

    Parameters:
        binary_img (ndarray): warped binary image of a sudoku square
        classifier (CompletedBoxClassifier): classifier for jigsaw borders
        debug (bool): debugging flag
        margin_pct (float): for each cell crop the margins based on this percentage,
            these margins are used for determining if a jigsaw border is placed
            on that side of the cell

    Returns:
        borders (list[list[CellBorders]]): matrix showing where the borders of the
            jigsaw regions are placed
    """
    box_len = int(binary_img.shape[0] / 9)

    borders = [[CellBorders() for i in range(9)] for j in range(9)]

    for i in range(9):
        for j in range(9):
            box = binary_img[
                i * box_len : (i + 1) * box_len - 1, j * box_len : (j + 1) * box_len - 1
            ].copy()
            margin = int(box_len * margin_pct)

            # up
            box_margin = box[:margin, :]
            predicted = classifier(box_margin)
            if predicted:
                borders[i][j].up = True
                if i > 0:
                    borders[i - 1][j].down = True

            # right
            box_margin = box[:, -margin:]
            predicted = classifier(box_margin)
            if predicted:
                borders[i][j].right = True
                if j < 8:
                    borders[i][j + 1].left = True

            # down
            box_margin = box[-margin:, :]
            predicted = classifier(box_margin)
            if predicted:
                borders[i][j].down = True
                if i < 8:
                    borders[i + 1][j].up = True

            # left
            box_margin = box[:, :margin]
            predicted = classifier(box_margin)
            if predicted:
                borders[i][j].left = True
                if j > 0:
                    borders[i][j - 1].right = True

    if debug:
        print("\n".join(["\t".join([str(x) for x in line]) for line in borders]))

    return borders


def fill_regions(borders):
    """Given the configuration of the jigsaw borders apply a recursive fill
    algorithm that determines and labels the jigsaw regions.

    Parameters:
        borders (list[list[CellBorders]]): configuration of the jigsaw borders

    Returns:
        regions_str (str): the region-labels of the cells (string representation)
    """
    zones = [[0] * 9 for _ in range(9)]

    def recursive_fill(i, j, val):
        if zones[i][j] != 0:
            return
        zones[i][j] = val
        if i > 0 and not borders[i][j].up:
            recursive_fill(i - 1, j, val)
        if i < 8 and not borders[i][j].down:
            recursive_fill(i + 1, j, val)
        if j > 0 and not borders[i][j].left:
            recursive_fill(i, j - 1, val)
        if j < 8 and not borders[i][j].right:
            recursive_fill(i, j + 1, val)

    curr = 0
    for i in range(9):
        for j in range(9):
            if zones[i][j] == 0:
                # found a new region
                curr += 1
                recursive_fill(i, j, curr)

    return "\n".join(["".join([str(x) for x in line]) for line in zones])


def find_colored_zones(img, binary_img, num_colors=3, debug=False, margin_pct=0.15):
    """Given the warped colored image of the jigsaw sudoku square, find the jigsaw
    regions based on the color of the cells. The color of a cell is approximated
    to the mean value of the pixels in that cell that are not part of a digit or
    the border of the cell (in order to ignore those we also use the binary
    representation of the sudoku square - we ignore the positions with white pixels)

    Parameters:
        img (ndarray): colored sudoku board
        binary_img (ndarray): binary representation of the board
        num_colors (int): number of distinct colors used for delimiting regions
        debug (bool): debugging flag
        margin_pct (float): for each cell crop the margins based on this percentage

    Returns:
        regions_str (str): the region-labels of the cells (string representation)
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    box_len = int(img.shape[0] / 9)
    colors = []

    fig = plt.figure(figsize=(9, 9))
    for i in range(9):
        for j in range(9):
            top = i * box_len
            bottom = (i + 1) * box_len
            left = j * box_len
            right = (j + 1) * box_len

            mean_color = np.array([0.0, 0.0, 0.0])
            mean_hue = np.array([0.0])
            cnt = 0
            for y in range(top, bottom):
                for x in range(left, right):
                    if binary_img[y, x] == 0:
                        mean_color += img[y, x]
                        mean_hue += hsv[y, x, 0]
                        cnt += 1
            mean_color = np.array(mean_color / cnt, dtype=np.uint8)
            mean_hue = mean_hue / cnt

            fig.add_subplot(9, 9, i * 9 + j + 1)
            plt.imshow([[mean_color[[2, 1, 0]]] * box_len] * box_len)
            plt.axis("off")

            # use the mean hue of the cell for color-based clustering of the cells
            colors.append(mean_hue)

    # use KMeans clustering to cluster the cells based on color
    clusters = list(KMeans(n_clusters=num_colors, random_state=42).fit_predict(colors))
    if debug:
        print(clusters)
    # set the color cluster for each cell
    zones = [[0] * 9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            zones[i][j] = clusters.pop(0) + 1

    def fill_conex_region(i, j, val):
        """Recursive fill for labelling conex regions with the same color"""
        color = zones[i][j]
        zones[i][j] = -val
        for d_i, d_j in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            ni, nj = i + d_i, j + d_j
            if 0 > ni or ni > 8 or 0 > nj or nj > 8 or zones[ni][nj] != color:
                continue
            fill_conex_region(ni, nj, val)

    # determine the conex regions with same color
    curr = 0
    for i in range(9):
        for j in range(9):
            if zones[i][j] < 0:
                continue
            curr += 1
            fill_conex_region(i, j, curr)

    ret_string = "\n".join(["".join([str(-x) for x in line]) for line in zones])
    if debug:
        plt.show()
        print(ret_string)
    else:
        fig.clear()
        plt.close(fig)

    return ret_string


def get_digit_cell(img, row, col, margin_pct=0.15):
    """For a given sudoku square, return the cropped cell at the specified row
    and column. Also crop the margins of that cell.

    Parameters:
        img (ndarray): image of sudoku square
        row (int): row of the returned cell (0 indexed)
        col (int): column of the returned cell (0 indexed)
        margin_pct (float): percentage of margin to be cropped

    Returns:
        cell (ndarray): the cropped cell
    """
    box_len = int(img.shape[0] / 9)
    margin = int(box_len * margin_pct)
    box = img[
        row * box_len : (row + 1) * box_len, col * box_len : (col + 1) * box_len
    ].copy()
    return box[margin:-margin, margin:-margin]


def count_difference(img1, img2, delta_x, delta_y):
    """Given two binary images count how many pixels do not match (considering
    an offset given for the first image).

    Parameters:
        img1 (ndarray): first binary image
        img2 (ndarray): second binary image
        delta_x (int): offset on the OX axis for the first image (in pixels)
        delta_y (int): offset on the OY axis for the first image (in pixels)

    Returns:
        cnt (int): number of mismatches
    """
    cnt = 0
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            nx = x + delta_x
            ny = y + delta_y
            if 0 > nx or nx >= img2.shape[1] or 0 > ny or ny >= img2.shape[0]:
                if img1[y, x] == 255:
                    cnt += 1
                continue
            if img1[y, x] != img2[ny, nx]:
                cnt += 1
    return cnt


def find_best_match(img, template, max_range=4):
    """Given a binary image and a template, find the offset that minimizes the
    number of mismatches between the two images. The offsets are searched in the
    [-max_range, max_range] interval.

    Returns the minimum number of mismatches computed.
    """
    best_cnt = float("inf")
    for delta_x in range(-max_range, max_range + 1):
        for delta_y in range(-max_range, max_range + 1):
            cnt = count_difference(img, template, delta_x, delta_y)
            best_cnt = min(best_cnt, cnt)
    return best_cnt


def find_best_template(img, templates, max_range=4):
    """Given an image and a set of templates, find the best matching template"""
    best_cnt = float("inf")
    best_template = None
    for i, template in enumerate(templates):
        cnt = find_best_match(img, template, max_range)
        if cnt < best_cnt:
            best_cnt = cnt
            best_template = i
    return best_template


def find_largest_polygons(binary_img, num=3):
    """Returns the `num` largest polygons (contours), based on area,
    found in the given binary image
    """
    polygons, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polygons = sorted(polygons, key=lambda p: -cv.contourArea(p))
    return polygons[:num]


def warp_faces_on_template(template, top, front, right):
    """Given the image of the template cube, warp the three sudokus on the
    corresponding faces

    The corners of the cube are considered constant (same template).
    """
    A = [5, 156]
    B = [297, 236]
    C = [297, 540]
    D = [3, 463]
    E = [274, 3]
    F = [566, 82]
    G = [566, 387]

    template = template.copy()

    # height and width of the faces
    h, w = top.shape[:2]
    face_pts = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])

    def warp_face(template, template_corners, face):
        """Warp the given face on top of the template in the quad described by
        the four `template_corners`
        """
        temp_pts = np.array(template_corners)
        # find the homography matrix for this transformation
        mat, _ = cv.findHomography(face_pts, temp_pts, cv.RANSAC, 5.0)
        # warp the face into the given shape inside an image with the same dimensions
        # as the template
        temp_warped = cv.warpPerspective(
            face, mat, (template.shape[1], template.shape[0])
        )

        # build a mask corresponding to the pixels where the warped sudoku will
        # be pasted
        mask = np.zeros(template.shape, dtype=np.uint8)
        cv.fillConvexPoly(mask, temp_pts, 255)
        mask = cv.bitwise_not(mask)

        # paste the warped sudoku on top of the template
        masked_template = cv.bitwise_and(template, mask)
        template = cv.bitwise_or(temp_warped, masked_template)
        return template

    # top
    template = warp_face(template, [E, F, B, A], top)
    # front
    template = warp_face(template, [A, B, C, D], front)
    # right
    template = warp_face(template, [B, F, G, C], right)

    return template

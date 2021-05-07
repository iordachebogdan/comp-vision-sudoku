import argparse
import os

import cv2 as cv
from tqdm import tqdm

from sudoku.generic_utils import load_images
from sudoku.image_processing import warp_faces_on_template
from sudoku.tasks import (
    make_prediction_task1,
    make_prediction_task2,
    make_prediction_task3,
)

parser = argparse.ArgumentParser()
parser.add_argument("task", help="which task to solve (1, 2 or 3)", type=int)
parser.add_argument(
    "input_dir", help="path to the directory containing the input images"
)
parser.add_argument(
    "output_dir",
    help="directory path for where to store the generated outputs (directory must exist)",
)
parser.add_argument(
    "--template-path", help="path to the sudoku cube template (required for task 3)"
)
parser.add_argument(
    "--digit-db-path",
    help="path to the directory containing the 9 digit templates (required for task 3)",
)
parser.add_argument(
    "--debug",
    help="display plots of intermediary steps, for debugging",
    action="store_true",
)


def solve_task1(all_images, all_filenames, output_dir):
    for img, filename in tqdm(zip(all_images, all_filenames), total=len(all_images)):
        try:
            predicted = make_prediction_task1(img.copy())
            with open(os.path.join(output_dir, f"{filename}_predicted.txt"), "w") as f:
                f.write(predicted)
        except Exception as ex:
            print(f"Failed on {filename}: {ex}")


def solve_task2(all_images, all_filenames, output_dir):
    for img, filename in tqdm(zip(all_images, all_filenames), total=len(all_images)):
        try:
            predicted = make_prediction_task2(img.copy())
            with open(os.path.join(output_dir, f"{filename}_predicted.txt"), "w") as f:
                f.write(predicted)
        except Exception as ex:
            print(f"Failed on {filename}: {ex}")


def solve_task3(all_images, all_filenames, output_dir, template_path, digit_db_path):
    # load the digit templates
    templates = []
    for i in range(1, 10):
        templates.append(
            cv.imread(os.path.join(digit_db_path, f"{i}.bmp"), cv.IMREAD_GRAYSCALE)
        )

    # load the cube template
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

    for img, filename in tqdm(zip(all_images, all_filenames), total=len(all_images)):
        try:
            predicted, faces = make_prediction_task3(img.copy(), templates)
            with open(os.path.join(output_dir, f"{filename}_predicted.txt"), "w") as f:
                f.write(predicted)

            # generate cube image
            result_img = warp_faces_on_template(template.copy(), *faces)
            cv.imwrite(os.path.join(output_dir, f"{filename}_result.jpg"), result_img)
        except Exception as ex:
            print(f"Failed on {filename}: {ex}")


def main():
    args = parser.parse_args()
    if args.task not in [1, 2, 3]:
        print("Invalid task number")
        return
    if args.task == 3 and (not args.template_path or not args.digit_db_path):
        print("You must specify the path to the sudoku cube template and digits db")
        return
    if not os.path.isdir(args.input_dir):
        print("Invalid input directory")
    if not os.path.isdir(args.output_dir):
        print("Invalid output directory")

    all_images, all_filenames = load_images(
        args.input_dir,
        resize=(720 if args.task != 3 else None),
        grayscale=(args.task == 3),
    )
    if args.task == 1:
        solve_task1(all_images, all_filenames, args.output_dir)
    elif args.task == 2:
        solve_task2(all_images, all_filenames, args.output_dir)
    else:
        solve_task3(
            all_images,
            all_filenames,
            args.output_dir,
            args.template_path,
            args.digit_db_path,
        )


if __name__ == "__main__":
    main()

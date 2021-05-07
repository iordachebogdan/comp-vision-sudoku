import argparse
import os
from tqdm import tqdm

from sudoku.generic_utils import load_images
from sudoku.tasks import make_prediction_task1

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


def main():
    args = parser.parse_args()
    if args.task not in [1, 2, 3]:
        print("Invalid task number")
        return
    if args.task == 3 and not args.template_path:
        print("You must specify the path to the sudoku cube template")
        return
    if not os.path.isdir(args.input_dir):
        print("Invalid input directory")
    if not os.path.isdir(args.output_dir):
        print("Invalid output directory")

    all_images, all_filenames = load_images(
        args.input_dir, resize=(720 if args.task != 3 else None)
    )
    if args.task == 1:
        solve_task1(all_images, all_filenames, args.output_dir)
    # elif args.task == 2:
    #     solve_task2(all_images, all_filenames)
    # else:
    #     solve_task3(all_images, all_filenames)


if __name__ == "__main__":
    main()

import argparse
import os
import pye57
import numpy as np
import glob
import ast

# fmt:off
def get_args():
    parser = argparse.ArgumentParser()
    assert(isinstance(parser, argparse.ArgumentParser))
    parser.add_argument('data_root', type=str, help='folder to collect, dataroot')
    parser.add_argument('write_path', type=str, help="folder to save .npy arrays")
    parser.add_argument('--label_literal', type=str, help="dict literal for turning file_class to label numbers")
    parser.add_argument('--text', action='store_true')
    return parser.parse_args()


# fmt:on
def annotation_to_7d(path, label):
    reader = pye57.E57(path, mode="r")
    all_points = None
    for i in range(reader.scan_count):
        scan = reader.read_scan(i, colors=True, ignore_missing_fields=True)
        points = np.column_stack((
            scan["cartesianX"],
            scan["cartesianY"],
            scan["cartesianZ"],
            scan["colorRed"],
            scan["colorGreen"],
            scan["colorBlue"],
            np.full((len(scan["cartesianX"])), label),
        ))

        if all_points is not None:
            all_points = np.concatenate((all_points, points))
        else:
            all_points = points
    return points


def main():
    args = get_args()
    # If folder, iterate over all files and add to list
    DATA_ROOT = args.data_root
    WRITE_PATH = args.write_path

    class2label = {"normal": 0, "plant": 1, "crack": 2, "spalling": 3}
    if args.label_literal is not None:
        class2label = ast.literal_eval(args.label_literal)

    # list all directories to list
    sample_dirs = [
        entry.path
        for entry in os.scandir(DATA_ROOT) if entry.is_dir()
    ]

    # Check and create dir
    os.makedirs(WRITE_PATH, exist_ok=True)

    # loop over directories' annotations folder using annotation to array
    for dir_path in sample_dirs:
        try:
            anno_path = os.path.join(dir_path, "labels")
            files = glob.glob(os.path.join(anno_path, "*.e57"))
            sample_points = None
            for f in files:
                file_class = os.path.basename(f).split("_")[0]

                if file_class not in class2label:
                    print(f"Skipped {f}: Class missing")
                    continue
                label = class2label[file_class]
                points = annotation_to_7d(f, label)

                sample_points = (
                    np.concatenate((sample_points, points))
                    if sample_points is not None
                    else points
                )

            write_file_path = os.path.join(WRITE_PATH, os.path.basename(dir_path))
            
            if sample_points.shape[0] <= 0:
                print(f"Skipped Sample at {dir_path}: No Points collected")
                continue

            if args.text:
                with open(write_file_path + ".txt", "w") as fout:
                    for i in range(sample_points.shape[0]):
                        fout.write(
                            f"{sample_points[i, 0]} {sample_points[i, 1]} {sample_points[i, 2]} {sample_points[i, 3]} {sample_points[i, 4]} {sample_points[i, 5]} {sample_points[i, 6]}\n"
                        )
            else:
                np.save(write_file_path, sample_points)
            
            print(f"Saved {write_file_path}")
        except Exception as e:
            print(f"Skipped Sample at {dir_path}: {e}")


if __name__ == "__main__":
    main()

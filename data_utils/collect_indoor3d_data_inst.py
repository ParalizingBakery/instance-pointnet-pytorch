import os
import sys
import argparse
from indoor3d_util import DATA_PATH, collect_point_label_inst

parser = argparse.ArgumentParser()
parser.add_argument('--text', help='save output as ascii text for debugging', action='store_true')
parser.add_argument('--out-dir', help='change output path', default='data/stanford_indoor3d')
parser.add_argument('--stats', action='store_true')
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d') if not args.out_dir else args.out_dir
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

file_format = 'numpy' if not args.text else 'txt'
extension = '.npy' if not args.text else '.txt'

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+extension # Area_1_hallway_1.npy
        collect_point_label_inst(anno_path, os.path.join(output_folder, out_filename), file_format)
    except:
        print(anno_path, 'ERROR!!')

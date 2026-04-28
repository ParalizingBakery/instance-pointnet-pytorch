"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.WPSSampleDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from sklearn.ensemble import IsolationForest
from ast import literal_eval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['normal', 'damage']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--model', type=str, help='name of model.py to use, enter name of file without extension')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or dimensions for testing')
    parser.add_argument('--init_num_classes', type=int, default=13, help='num classes as model init argument')
    parser.add_argument('--root', type=str, default='data/stanford_indoor3d/')
    parser.add_argument('--struct_min', type=str, default='None', help='list literal for structure coordinates ex. "[Xmin, Ymin, Zmin]"')
    parser.add_argument('--struct_max', type=str, default='None')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = args.num_classes

    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.root
    struct_min = literal_eval(args.struct_min)
    struct_max = literal_eval(args.struct_max)

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(
        root,
        block_points=NUM_POINT,
        struct_coord_min=struct_min,
        struct_coord_max=struct_max,
    )
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = args.model
    if args.model is None:
        # Insane logic: model name from file order
        model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(args.init_num_classes).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    test_sim = True

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            # if args.visual:
            #     fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
            #     fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0
                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    # returns (B,N,D)
                    seg_pred, embeddings = classifier(torch_data)
                    assert isinstance(embeddings, torch.Tensor)

                    # Remove Filler batches to match batch_size
                    embeddings = embeddings[batch_smpw != 0]

                    # Set contamination to 0.1 for isolating top 10% points in predict,
                    clf = IsolationForest(
                        n_estimators=200,
                        max_samples=1024,
                        contamination='auto',
                        max_features=0.5, # all feats
                        random_state=4098,
                    )
                    embed_cpu = embeddings.cpu()
                    # merge batches, turn to shape (B*N, D)
                    
                    clf.fit(embed_cpu)
                    # outputs -1 for anomalies, 1 for normal
                    anomaly_pred = clf.predict(embed_cpu)
                    # normal label is 0 and damage is 1
                    anomaly_pred[anomaly_pred == 1] = 0
                    anomaly_pred[anomaly_pred == -1] = 1
                    print(anomaly_pred.shape)
                    # if (scene_id[batch_idx] == '0000'):
                        # np.savetxt(f'0000_batch_channels.txt', np.column_stack((batch_data[batch_smpw!=0].reshape(-1, 9), anomaly_pred)))

                    # change back to [B,N], with only non-filler batches.
                    batch_pred_label = anomaly_pred.reshape(-1, NUM_POINT)
                    # batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
            
            l = 1
            pos = pred_label == l
            neg = pred_label != l
            anomaly = whole_scene_label == l
            normal = whole_scene_label == 0
            # total = anomaly | normal
            true_pos = pos & anomaly
            false_pos = pos & normal
            # true_neg = neg & normal
            # false_neg = neg & anomaly
            total_seen_class_tmp[l] += np.sum(anomaly)
            total_correct_class_tmp[l] += np.sum(true_pos)
            total_iou_deno_class_tmp[l] += np.sum((pos | anomaly))
            total_seen_class[l] += total_seen_class_tmp[l]
            total_correct_class[l] += total_correct_class_tmp[l]
            total_iou_deno_class[l] += total_iou_deno_class_tmp[l]
            precision_tmp = np.sum(true_pos) / max(np.sum(pos), 1e-8)
            recall_tmp = np.sum(true_pos) / max(np.sum(anomaly), 1e-8)
            f1_tmp = 2.0 * (precision_tmp * recall_tmp) / (precision_tmp + recall_tmp)
            fpr_tmp = np.sum(false_pos) / np.sum(normal)

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            log_string(f'Precision: {precision_tmp}')
            log_string(f'Recall: {recall_tmp}')
            log_string(f'F1: {f1_tmp}')
            log_string(f'FPR: {fpr_tmp}')
            log_string(f'csv: {tmp_iou}, {precision_tmp}, {recall_tmp}, {f1_tmp}, {fpr_tmp}')
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()

            if args.visual:
                np.savetxt(
                    os.path.join(visual_dir, f"{scene_id[batch_idx]}_pred.txt"),
                    np.column_stack(
                        (whole_scene_data.reshape(-1, 6), pred_label)
                    ),
                )
                np.savetxt(
                    os.path.join(visual_dir, f"{scene_id[batch_idx]}_gt.txt"),
                    np.column_stack(
                        (whole_scene_data.reshape(-1, 6), whole_scene_label)
                    ),
                )

                # color = g_label2color[pred_label[i]]
                # color_gt = g_label2color[whole_scene_label[i]]
                # if args.visual:
                #     fout.write('v %f %f %f %d %d %d\n' % (
                #         whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                #         color[2]))
                #     fout_gt.write(
                #         'v %f %f %f %d %d %d\n' % (
                #             whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                #             color_gt[1], color_gt[2]))

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(1, 2):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        # log_string('eval point avg class IoU: %f' % np.mean(IoU))
        # log_string('eval whole scene point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
        # log_string('eval whole scene point accuracy: %f' % (
        #         np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)

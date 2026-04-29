"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader_inst import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
#            'board', 'clutter']
# class2label = {cls: i for i, cls in enumerate(classes)}
# seg_classes = class2label
# seg_label_to_cat = {}
# for i, cat in enumerate(seg_classes.keys()):
# seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--pos_strategy', type=str, default='easy', help='pos_strategy string')
    parser.add_argument('--neg_strategy', type=str, default='semihard', help='neg_strategy string')
    parser.add_argument('--margin', type=float, default=0.05, help='margin for triplet loss')
    parser.add_argument('--test_sample_rate', type=float, default=0.25, help='nultiplier for number of points in embedding evaluation')
    parser.add_argument('--eval_num_points', type=float, default=2048, help='points to randomly select for each sample eval')
    parser.add_argument('--save_epoch', type=int, default=3, help='save every n epochs')
    parser.add_argument('--data_root', type=str, default='data/stanford_indoor3d_inst/', help='location of collected data')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_root
    NUM_DIMENSIONS = 128
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=args.test_sample_rate, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_DIMENSIONS).cuda()
    # miner and loss using cpu is faster 1.6it/s from 1.3it/s using cuda
    # probably because they are called per sample
    criterion = losses.TripletMarginLoss(margin = args.margin)
    miner_func = miners.BatchEasyHardMiner(pos_strategy=args.pos_strategy, neg_strategy=args.neg_strategy)

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    highest_mapr = 0.0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target, room_idx) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # output of model is [B, N, D]
            seg_pred, trans_feat = classifier(points)
            # seg_pred = seg_pred.contiguous().view(-1, NUM_DIMENSIONS)

            # Instance Labels are per-room, not global across 6 areas
            # One way to optimise is to make mining and loss parallel across samples
            batch_loss_sum = torch.tensor(0.0).cuda()
            for sample_idx in range(seg_pred.shape[0]):
                # Mine triplets, default pos=easy and neg=semihard
                sample_embed = seg_pred[sample_idx]
                sample_target = target[sample_idx]
                miner_output = miner_func(sample_embed, sample_target)
                batch_loss_sum += criterion(sample_embed, sample_target, miner_output)

            loss = batch_loss_sum / args.batch_size
            loss.backward()
            optimizer.step()

            # pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            # correct = np.sum(pred_choice == batch_label)
            # total_correct += correct
            # total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        # log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % args.save_epoch == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + f'/model_{epoch}.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        SAMPLE_POINTS = args.eval_num_points
        FAISS_GPU_POINTS = 2048
        MAX_INST = 500
        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            loss_sum = 0.0
            mapr_sum = 0.0
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target, room_idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = (
                    points.float().cuda(),
                    target.long().cuda().contiguous(),
                )
                points = points.transpose(2, 1)

                embed_pred, trans_feat = classifier(points)
                SAMPLES_PER_CHUNK = int(FAISS_GPU_POINTS / SAMPLE_POINTS)

                batch_loss_sum = torch.tensor(0.0).cuda()
                batch_mapr_sum = torch.tensor(0.0)

                # Generate subsample for each batch of size 512, [B, EVAL_POINTS]
                all_indices = torch.stack(
                    [
                        torch.randperm(NUM_POINT, device=embed_pred.device)[
                            :SAMPLE_POINTS
                        ]
                        for _ in range(BATCH_SIZE)
                    ]
                )

                for start_chunk in range(0, BATCH_SIZE, SAMPLES_PER_CHUNK):
                    end_chunk = min(BATCH_SIZE, start_chunk + SAMPLES_PER_CHUNK)
                    chunk_indices = all_indices[start_chunk:end_chunk]

                    # Expand to [SELECTED_SAMPLES, EVAL_POINTS, D]
                    chunk_embed_indices = chunk_indices.unsqueeze(-1).expand(
                        -1, -1, NUM_DIMENSIONS
                    )

                    chunk_embed = (
                        torch.gather(
                            embed_pred[start_chunk:end_chunk], 1, chunk_embed_indices
                        )
                        .reshape(-1, NUM_DIMENSIONS)
                        .contiguous()
                    )

                    # Add an Offset to each room so instances from same room are the same
                    chunk_target = torch.gather(
                        target[start_chunk:end_chunk], 1, chunk_indices
                    ).contiguous()
                    room_offsets = (
                        room_idx[start_chunk:end_chunk].to(chunk_target.device)
                        * MAX_INST
                    )
                    chunk_target = (
                        (chunk_target + room_offsets.unsqueeze(1)).view(-1).contiguous()
                    )

                    calculator = AccuracyCalculator(
                        include=tuple(["mean_average_precision_at_r"]),
                        k=None,
                        avg_of_avgs=True,
                        device=chunk_embed.device,
                    )
                    results = calculator.get_accuracy(
                        query=chunk_embed,
                        query_labels=chunk_target,
                        reference=chunk_embed,
                        reference_labels=chunk_target,
                    )
                    batch_mapr_sum += (
                        results["mean_average_precision_at_r"] * SAMPLES_PER_CHUNK
                    )

                mapr = batch_mapr_sum / BATCH_SIZE
                mapr_sum += mapr

            mean_mapr = mapr_sum / float(num_batches)
            log_string('eval mean mapr: %f' % (mean_mapr))

            if mean_mapr >= highest_mapr:
                highest_mapr = mean_mapr
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'eval_mapr': mean_mapr,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string(f'Saving model epoch {epoch} with mapr {mean_mapr}')
            log_string('highest_mapr: %f' % highest_mapr)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)

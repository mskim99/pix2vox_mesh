# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import random

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
from torch.autograd import Variable

import utils.binvox_visualization
import utils.binvox_rw
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.generator import Generator

import joblib

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             generator=None,
             volume_scaler=None,
             image_scaler=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if generator is None:
        generator = Generator(cfg)

        if torch.cuda.is_available():
            generator = torch.nn.DataParallel(generator).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        generator.load_state_dict(checkpoint['generator_state_dict'])

    if volume_scaler is None or image_scaler is None:
        volume_scaler = joblib.load('./output/logs/checkpoints2/volume_scaler.pkl')
        image_scaler = joblib.load('./output/logs/checkpoints2/image_scaler.pkl')

    # Set up loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()
    # ce_loss = torch.nn.CrossEntropyLoss()
    l1_loss = torch.nn.L1Loss()
    # smooth_l1_loss = torch.nn.SmoothL1Loss()
    # huber_loss = torch.nn.HuberLoss(delta=0.5)

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    L1_losses = utils.network_utils_GAN.AverageMeter()

    # Switch models to evaluation mode
    generator.eval()

    n_batches = len(test_data_loader)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volumes, ground_truth_mesh) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():

            rendering_images = rendering_images / 255.

            ground_truth_mesh = ground_truth_mesh.float()
            ground_truth_mesh = ground_truth_mesh.view([1, 3, -1])
            gtm_vertice_num = ground_truth_mesh.shape[2]

            if gtm_vertice_num != 1024:
                v_sub = 1024 - gtm_vertice_num
                if v_sub > 0:
                    add_vector = torch.zeros([1, 3, v_sub])
                    ground_truth_mesh = torch.cat((ground_truth_mesh, add_vector), dim=2)
                else:
                    ground_truth_mesh = ground_truth_mesh[:, :, 0:1024]

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_mesh = utils.network_utils.var_or_cuda(ground_truth_mesh)

            gen_mesh = generator(rendering_images)

            L2_loss = mse_loss(gen_mesh, ground_truth_mesh)
            L1_loss = l1_loss(gen_mesh, ground_truth_mesh)

            # Append loss to average metrics
            L1_losses.update(L1_loss.item())

            # Volume Visualization
            '''
            gv = gen_volumes.cpu().numpy()            
            np.save('./output/voxel2/gv/gv_' + str(sample_idx).zfill(6) + '.npy', gv)
            gtv = ground_truth_volumes.cpu().numpy()
            np.save('./output/voxel2/gtv/gtv_' + str(sample_idx).zfill(6) + '.npy', gtv)
            '''

            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s L1Loss = %.6f'
                  % (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, L1_loss.item()))

    print('[INFO] %s Test[%d] Loss Mean / L1Loss = %.6f'
          % (dt.now(), n_samples, L1_losses.avg))

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Generator/L1Loss', L1_losses.avg, epoch_idx)

    return 0.0 # t = 0.40
    # return min_loss

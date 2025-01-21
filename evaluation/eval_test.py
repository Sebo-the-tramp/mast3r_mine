#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------

import sys 
import torch
import numpy as np

from collections import defaultdict

sys.path.append('/home/sebastian.cavada/scsv/thesis/mast3r_mine')
sys.path.append('/home/sebastian.cavada/scsv/thesis/mast3r_mine/dust3r')

import mast3r.utils.path_to_dust3r  # noqai
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa

from dust3r.training import * # noqai

from mast3r.model import AsymmetricMASt3R
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, Regr3D, InfoNCE, Regr3D_ScaleShiftInv, ParamLoss, ReprojectionLoss
from mast3r.datasets import ARKitScenes, BlendedMVS, Co3d, MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD

import mast3r.utils.path_to_dust3r  # noqa
# add mast3r classes to dust3r imports
import dust3r.training
dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.training.Regr3D = Regr3D
dust3r.training.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss
dust3r.training.ParamLoss = ParamLoss
dust3r.training.ReprojectionLoss = ReprojectionLoss

import dust3r.datasets
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD

from dust3r.losses import Sum
from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud

from typing import Sized

# VARIABLES
norm_mode = "avg_dis"
gt_scale = False
sky_loss_value=0
max_metric_scale=False
loss_in_log=False
norm_all=False
reduction = 'mean'

# CODE

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

def get_all_pts3d(gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        # if loss_in_log == 'before':
        #     # this only make sense when depth_mode == 'linear'
        #     gt_pts1 = apply_log_to_norm(gt_pts1)
        #     gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not norm_all:
            if max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], norm_mode,
                                                                valid1[mask], valid2[mask])

        if norm_mode and not gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

def loss_L21(a, b):
    assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
    dist = distance(a, b)
    assert dist.ndim == a.ndim - 1  # one dimension less
    if reduction == 'none':
        return dist
    if reduction == 'sum':
        return dist.sum()
    if reduction == 'mean':
        return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
    raise ValueError(f'bad {reduction=} mode')

def distance(a, b):
    return torch.norm(a - b, dim=-1)  # normalized L2 distance


def compute_metrics(gt1, gt2, pred1, pred2, **kw):
    # I guess that here the loss is computed for each 3D point in each view
    # this is great because we can have a loss for each point
    
    gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
        get_all_pts3d(gt1, gt2, pred1, pred2, **kw)    
    
    # print(gt_pts1, gt_pts2)
    # print(pred_pts1, pred_pts2)

    if sky_loss_value > 0:
        assert reduction == 'none', 'sky_loss_value should be 0 if no conf loss'        
        # add the sky pixel as "valid" pixels...
        mask1 = mask1 | sky1
        mask2 = mask2 | sky2

    pred_pts1 = pred_pts1[mask1]
    gt_pts1 = gt_pts1[mask1]    
    loss_img1 = loss_L21(pred_pts1, gt_pts1)

    pred_pts2 = pred_pts2[mask2]
    gt_pts2 = gt_pts2[mask2]
    loss_img2 = loss_L21(pred_pts2, gt_pts2)
    
    if sky_loss_value > 0:
        assert reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
        # ... but force the loss to be high there
        loss_img1 = torch.where(sky1[mask1], sky_loss_value, loss_img1)
        loss_img2 = torch.where(sky2[mask2], sky_loss_value, loss_img2)
            
    self_name = "loss_l21"
    details = {self_name + '_pts3d_1': float(loss_img1.mean()), self_name + '_pts3d_2': float(loss_img2.mean())} 
    return Sum((loss_img1, mask1), (loss_img2, mask2)), (details | monitoring)

    
    # # loss on img1 side
    # l1 = criterion(pred_pts1[mask1], gt_pts1[mask1])
    # # loss on gt2 side
    # l2 = criterion(pred_pts2[mask2], gt_pts2[mask2])
    # self_name = type(self).__name__
    # details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
    # return Sum((l1, mask1), (l2, mask2)), (details | monitoring)

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res

def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def test_one_batch(batch, model, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.amp.autocast('cuda', enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        loss, details = compute_metrics(view1, view2, pred1, pred2)

        # print(details)

    return loss, details

        # # loss is supposed to be symmetric
        # with torch.cuda.amp.autocast(enabled=False):
        #     loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    # result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    # return result[ret] if ret else result


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module,
                   data_loader: Sized, epoch: int,
                   print_freq = 2):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)   

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):                

        loss_value, loss_details = test_one_batch(batch, model, 'cuda', symmetrize_batch=True, use_amp=False, ret='loss')

        # loss_value, loss_details = loss_details  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # aggs = [('avg', 'global_avg'), ('med', 'median')]
    # results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    # if log_writer is not None:
    #     for name, val in results.items():
    #         log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    # return results

def main():

    # fix the seed
    seed = 777 + misc.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_string = "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), use_intrinsics=True, use_extrinsics=True)"
    model = eval(model_string)
    model.to('cuda')

    test_dataset = "1_000 @ ARKitScenes(split='test', aug_crop=256, resolution=(512, 384), transform=ColorJitter, ROOT='/data3/sebastian.cavada/datasets/arkitscenes_test', seed=777, n_corres=1024)"
    print('Building test dataset {:s}'.format(test_dataset))

    batch_size = 16
    num_workers = 8

    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, batch_size, num_workers, test=True)
                        for dataset in test_dataset.split('+')}
    
    for test_name, testset in data_loader_test.items():        

        stats = test_one_epoch(model, testset, epoch=0)
        # test_stats[test_name] = stats

        # # Save best of all
        # if stats['loss_med'] < best_so_far:
        #     best_so_far = stats['loss_med']
        #     new_best = True


if __name__ == "__main__":
    main()
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of MASt3R training losses
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from dust3r.losses import Regr3D as Regr3D_dust3r
from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz


class Regr3D (Regr3D_dust3r):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, opt_fit_gt=False,
                 sky_loss_value=2, max_metric_scale=False, loss_in_log=False):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
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

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != 'before':
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != 'before':
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth scale.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


class MatchingCriterion(BaseCriterion):
    def __init__(self, reduction='mean', fp=torch.float32):
        super().__init__(reduction)
        self.fp = fp

    def forward(self, a, b, valid_matches=None, euc=False):
        assert a.ndim >= 2 and 1 <= a.shape[-1], f'Bad shape = {a.shape}'
        dist = self.loss(a.to(self.fp), b.to(self.fp), valid_matches, euc=euc)
        # one dimension less or reduction to single value
        assert (valid_matches is None and dist.ndim == a.ndim -
                1) or self.reduction in ['mean', 'sum', '1-mean', 'none']
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if self.reduction == '1-mean':
            return 1. - dist.mean() if dist.numel() > 0 else dist.new_ones(())
        raise ValueError(f'bad {self.reduction=} mode')

    def loss(self, a, b, valid_matches=None):
        raise NotImplementedError


class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode='all', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        assert valid_matches.shape == torch.Size([B, N]) 
        # assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0

        # No valid matches -> We need to decide what to do TODO
        if valid_matches.sum() == 0:
            # return None
            return torch.tensor(0.0, device=desc1.device)

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':            # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]


class APLoss (MatchingCriterion):
    """ AP loss.
    """

    def __init__(self, nq='torch', min=0, max=1, euc=False, **kw):
        super().__init__(**kw)
        # Exact/True AP loss (not differentiable)
        if nq == 0:
            nq = 'sklearn'  # special case
        try:
            self.compute_AP = eval('self.compute_true_AP_' + nq)
        except:
            raise ValueError("Unknown mode %s for AP loss" % nq)

    @staticmethod
    def compute_true_AP_sklearn(scores, labels):
        def compute_AP(label, score):
            return average_precision_score(label, score)

        aps = scores.new_zeros((scores.shape[0], scores.shape[1]))
        label_np = labels.cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        for bi in range(scores_np.shape[0]):
            for i in range(scores_np.shape[1]):
                labels = label_np[bi, i, :]
                if labels.sum() < 1:
                    continue
                aps[bi, i] = compute_AP(labels, scores_np[bi, i, :])
        return aps

    @staticmethod
    def compute_true_AP_torch(scores, labels):
        assert scores.shape == labels.shape
        B, N, M = labels.shape
        dev = labels.device
        with torch.no_grad():
            # sort scores
            _, order = scores.sort(dim=-1, descending=True)
            # sort labels accordingly
            labels = labels[torch.arange(B, device=dev)[:, None, None].expand(order.shape),
                            torch.arange(N, device=dev)[None, :, None].expand(order.shape),
                            order]
            # compute number of positives per query
            npos = labels.sum(dim=-1)
            assert torch.all(torch.isclose(npos, npos[0, 0])
                             ), "only implemented for constant number of positives per query"
            npos = int(npos[0, 0])
            # compute precision at each recall point
            posrank = labels.nonzero()[:, -1].view(B, N, npos)
            recall = torch.arange(1, 1 + npos, dtype=torch.float32, device=dev)[None, None, :].expand(B, N, npos)
            precision = recall / (1 + posrank).float()
            # average precision values at all recall points
            aps = precision.mean(dim=-1)

        return aps

    def loss(self, desc1, desc2, valid_matches=None, euc=False):  # if matches is None, positives are the diagonal
        B, N1, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2

        scores = get_similarities(desc1, desc2, euc)

        labels = torch.zeros([B, N1, N2], dtype=scores.dtype, device=scores.device)

        # allow all diagonal positives and only mask afterwards
        labels.diagonal(dim1=-2, dim2=-1)[...] = 1.
        apscore = self.compute_AP(scores, labels)
        if valid_matches is not None:
            apscore = apscore[valid_matches]
        return apscore


class MatchingLoss (Criterion, MultiLoss):
    """ 
    Matching loss per image 
    only compare pixels inside an image but not in the whole batch as what would be done usually
    """

    def __init__(self, criterion, withconf=False, use_pts3d=False, negatives_padding=0, blocksize=4096):
        super().__init__(criterion)
        self.negatives_padding = negatives_padding
        self.use_pts3d = use_pts3d
        self.blocksize = blocksize
        self.withconf = withconf

    def add_negatives(self, outdesc2, desc2, batchid, x2, y2):
        if self.negatives_padding:
            B, H, W, D = desc2.shape
            negatives = torch.ones([B, H, W], device=desc2.device, dtype=bool)
            negatives[batchid, y2, x2] = False
            sel = negatives & (negatives.view([B, -1]).cumsum(dim=-1).view(B, H, W)
                               <= self.negatives_padding)  # take the N-first negatives
            outdesc2 = torch.cat([outdesc2, desc2[sel].view([B, -1, D])], dim=1)
        return outdesc2

    def get_confs(self, pred1, pred2, sel1, sel2):
        if self.withconf:
            if self.use_pts3d:
                outconfs1 = pred1['conf'][sel1]
                outconfs2 = pred2['conf'][sel2]
            else:
                outconfs1 = pred1['desc_conf'][sel1]
                outconfs2 = pred2['desc_conf'][sel2]
        else:
            outconfs1 = outconfs2 = None
        return outconfs1, outconfs2

    def get_descs(self, pred1, pred2):
        if self.use_pts3d:
            desc1, desc2 = pred1['pts3d'], pred2['pts3d_in_other_view']
        else:
            desc1, desc2 = pred1['desc'], pred2['desc']
        return desc1, desc2

    def get_matching_descs(self, gt1, gt2, pred1, pred2, **kw):
        outdesc1 = outdesc2 = outconfs1 = outconfs2 = None
        # Recover descs, GT corres and valid mask
        desc1, desc2 = self.get_descs(pred1, pred2)

        (x1, y1), (x2, y2) = gt1['corres'].unbind(-1), gt2['corres'].unbind(-1)
        valid_matches = gt1['valid_corres']

        # Select descs that have GT matches
        B, N = x1.shape
        batchid = torch.arange(B)[:, None].repeat(1, N)  # B, N
        outdesc1, outdesc2 = desc1[batchid, y1, x1], desc2[batchid, y2, x2]  # B, N, D

        # Padd with unused negatives
        outdesc2 = self.add_negatives(outdesc2, desc2, batchid, x2, y2)

        # Gather confs if needed
        sel1 = batchid, y1, x1
        sel2 = batchid, y2, x2
        outconfs1, outconfs2 = self.get_confs(pred1, pred2, sel1, sel2)

        return outdesc1, outdesc2, outconfs1, outconfs2, valid_matches, {'use_euclidean_dist': self.use_pts3d}

    def blockwise_criterion(self, descs1, descs2, confs1, confs2, valid_matches, euc, rng=np.random, shuffle=True):
        loss = None
        details = {}
        B, N, D = descs1.shape

        if N <= self.blocksize:  # Blocks are larger than provided descs, compute regular loss
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
        else:  # Compute criterion on the blockdiagonal only, after shuffling
            # Shuffle if necessary
            matches_perm = slice(None)
            if shuffle:
                matches_perm = np.stack([rng.choice(range(N), size=N, replace=False) for _ in range(B)])
                batchid = torch.tile(torch.arange(B), (N, 1)).T
                matches_perm = batchid, matches_perm

            descs1 = descs1[matches_perm]
            descs2 = descs2[matches_perm]
            valid_matches = valid_matches[matches_perm]

            assert N % self.blocksize == 0, "Error, can't chunk block-diagonal, please check blocksize"
            n_chunks = N // self.blocksize
            descs1 = descs1.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            descs2 = descs2.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            valid_matches = valid_matches.view([B * n_chunks, self.blocksize])
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
            if self.withconf:
                confs1, confs2 = map(lambda x: x[matches_perm], (confs1, confs2))  # apply perm to confidences if needed

        if self.withconf:
            # split confidences between positives/negatives for loss computation
            details['conf_pos'] = map(lambda x: x[valid_matches.view(B, -1)], (confs1, confs2))
            details['conf_neg'] = map(lambda x: x[~valid_matches.view(B, -1)], (confs1, confs2))
            details['Conf1_std'] = confs1.std()
            details['Conf2_std'] = confs2.std()

        return loss, details

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Gather preds and GT
        descs1, descs2, confs1, confs2, valid_matches, monitoring = self.get_matching_descs(
            gt1, gt2, pred1, pred2, **kw)

        # loss on matches
        loss, details = self.blockwise_criterion(descs1, descs2, confs1, confs2,
                                                 valid_matches, euc=monitoring.pop('use_euclidean_dist', False))

        details[type(self).__name__] = float(loss.mean())
        return loss, (details | monitoring)

class ConfMatchingLoss(ConfLoss):
    """ Weight matching by learned confidence. Same as ConfLoss but for a matching criterion
        Assuming the input matching_loss is a match-level loss.
    """

    def __init__(self, pixel_loss, alpha=1., confmode='prod', neg_conf_loss_quantile=False):
        super().__init__(pixel_loss, alpha)
        self.pixel_loss.withconf = True
        self.confmode = confmode
        self.neg_conf_loss_quantile = neg_conf_loss_quantile

    def aggregate_confs(self, confs1, confs2):  # get the confidences resulting from the two view predictions
        if self.confmode == 'prod':
            confs = confs1 * confs2 if confs1 is not None and confs2 is not None else 1.
        elif self.confmode == 'mean':
            confs = .5 * (confs1 + confs2) if confs1 is not None and confs2 is not None else 1.
        else:
            raise ValueError(f"Unknown conf mode {self.confmode}")
        return confs

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        loss, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # Recover confidences for positive and negative samples
        conf1_pos, conf2_pos = details.pop('conf_pos')
        conf1_neg, conf2_neg = details.pop('conf_neg')
        conf_pos = self.aggregate_confs(conf1_pos, conf2_pos)

        # weight Matching loss by confidence on positives
        conf_pos, log_conf_pos = self.get_conf_log(conf_pos)
        conf_loss = loss * conf_pos - self.alpha * log_conf_pos
        
        # average + nan protection (in case of no valid pixels at all)
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
        # Add negative confs loss to give some supervision signal to confidences for pixels that are not matched in GT
        if self.neg_conf_loss_quantile:
            conf_neg = torch.cat([conf1_neg, conf2_neg])
            conf_neg, log_conf_neg = self.get_conf_log(conf_neg)

            # recover quantile that will be used for negatives loss value assignment
            neg_loss_value = torch.quantile(loss, self.neg_conf_loss_quantile).detach()
            neg_loss = neg_loss_value * conf_neg - self.alpha * log_conf_neg

            neg_loss = neg_loss.mean() if neg_loss.numel() > 0 else 0
            conf_loss = conf_loss + neg_loss

        return conf_loss, dict(matching_conf_loss=float(conf_loss), **details)


class ParamLoss(MultiLoss):
    """
    Computes a loss between real and predicted camera intrinsics and extrinsics.
    Handles differences in value ranges by applying logarithmic scaling to each parameter type.
    """
    def __init__(self, intrinsic_weight=1.0, extrinsic_weight=1.0):
        super(ParamLoss, self).__init__()
        self.intrinsic_weight = intrinsic_weight
        self.extrinsic_weight = extrinsic_weight

    def log_loss(self, real, pred):
        """
        Compute the logarithmic loss to handle scale differences effectively.
        """
        # Use log1p (log(1 + x)) for numerical stability
        # return torch.mean((torch.log1p(real) - torch.log1p(pred))**2)
        real = torch.log1p(torch.nn.functional.softplus(real))
        pred = torch.log1p(torch.nn.functional.softplus(pred))
        return torch.mean((real - pred) ** 2)

    def compute_loss(self, gt1, gt2, pred1, pred2):
        """
        Compute the total loss with separate contributions from intrinsics and extrinsics.
        """

        # extracting ground_truth and setting to the correct values
        ext1_gt = gt1['camera_pose'].flatten(1)
        ext2_gt = gt2['camera_pose'].flatten(1)
        int1_gt = gt1['camera_intrinsics'].flatten(1)[:,:6]
        int2_gt = gt2['camera_intrinsics'].flatten(1)[:,:6]

        pred_extrinsics1 = pred1['camera_pose']
        pred_extrinsics2 = pred2['camera_pose']
        pred_intrinsics1 = pred1['camera_intrinsics']
        pred_intrinsics2 = pred2['camera_intrinsics']

        intrinsic_loss_view_1 = self.log_loss(int1_gt, pred_intrinsics1)
        intrinsic_loss_view_2 = self.log_loss(int2_gt, pred_intrinsics2)

        extrinsic_loss_view_1 = self.log_loss(ext1_gt, pred_extrinsics1)
        extrinsic_loss_view_2 = self.log_loss(ext2_gt, pred_extrinsics2)        

        total_loss_intrinsics = intrinsic_loss_view_1 + intrinsic_loss_view_2
        total_loss_extrinsics = extrinsic_loss_view_1 + extrinsic_loss_view_2

        total_loss = self.intrinsic_weight * total_loss_intrinsics + self.extrinsic_weight * total_loss_extrinsics

        details = {
            "ParamLoss_intrinsic": float(total_loss_intrinsics),
            "ParamLoss_extrinsic": float(total_loss_extrinsics),
        }
        return total_loss, details

    def get_name(self):
        return "ParamLoss"
    

class ReprojectionLoss(Criterion, MultiLoss):
    """
    Computes a loss between real and predicted camera intrinsics and extrinsics.
    Handles differences in value ranges by applying logarithmic scaling to each parameter type.
    """
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, sky_loss_value=2, max_metric_scale=False):
        super().__init__(criterion)
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode

        self.loss_in_log = False
        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale
        self.gt_scale = gt_scale
        self.threshold = 1.5

    def project_points(self, points, intrinsics, extrinsics_relative_cam1):        

        R_batch = extrinsics_relative_cam1[:, :3, :3]
        T_batch = extrinsics_relative_cam1[:, :3, 3]

        # Computes the 3D points transformation batched
        pts3d_transformed = torch.einsum(
            "bij,bhwj->bhwi", R_batch, points - T_batch[:, None, None, :]
        )

        # Extract the intrinsic parameters for each batch        
        f_x = intrinsics[:, 0].unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
        f_y = intrinsics[:, 4].unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
        c_x = intrinsics[:, 2].unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
        c_y = intrinsics[:, 5].unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)

        # project points to 2D
        x = pts3d_transformed[..., 0]  # Shape: (B, H, W)
        y = pts3d_transformed[..., 1]  # Shape: (B, H, W)
        z = pts3d_transformed[..., 2]  # Shape: (B, H, W)        

        H = pts3d_transformed.shape[1]
        W = pts3d_transformed.shape[2]

        # normalize to have everything in the range of the [-1, 1]
        f_x = f_x * 2 / W
        f_y = f_y * 2 / H
        c_x = (c_x * 2 / W) - 1
        c_y = (c_y * 2 / H) - 1

        u = (f_x * x / z) + c_x
        v = (f_y * y / z) + c_y

        projected_grid = torch.stack([u, v], dim=-1)          
        mask = (projected_grid[..., 0] > -self.threshold) & (projected_grid[..., 0] < self.threshold) & \
            (projected_grid[..., 1] > -self.threshold) & (projected_grid[..., 1] < self.threshold)

        return  projected_grid, mask
    
    def reprojection_loss(self, pts2d, valid):

        # what if there is a batch with different orientations? shit
        B, H, W, _ = pts2d.shape
        
        gt = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H, device=pts2d.device),
            torch.linspace(-1, 1, W, device=pts2d.device),
            indexing='ij'
        ), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)   

        pts2d_valid = pts2d[valid]
        gt = gt[valid]

        # Compute the L2 distance between the predicted and ground truth points
        loss = self.criterion(pts2d_valid, gt)
        return loss

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
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

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}
    

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):

        _, _, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        intrinsics_1 = gt1['camera_intrinsics'].flatten(1)[:,:6]
        intrinsics_2 = gt2['camera_intrinsics'].flatten(1)[:,:6]

        # since they are symmetric the same extrinsics will be found pairwise e.g.
        # extrinsics_1 = [A, B, C, D] -> extrinsics_2 = [B, A, D, C]        
        extrinsics_1 = gt1['camera_pose'][::2,:,:]
        extrinsics_2 = gt2['camera_pose'][::2,:,:]

        # print("IMAGE PAIR RESPONSIBLE:", gt1['label'])
        
        # repeat as the batch size B
        extrinsics_in_same_view = torch.eye(4).to(extrinsics_1.device).reshape(1, 4, 4).repeat(pred_pts1.shape[0], 1, 1)                

        world_to_cam1 = torch.inverse(extrinsics_1)
        extrinsics_2_relative_cam1 = torch.matmul(world_to_cam1, extrinsics_2)

        world_to_cam2 = torch.inverse(extrinsics_2)
        extrinsics_1_relative_cam2 = torch.matmul(world_to_cam2, extrinsics_1)

        # here the order matter, as per my calculation they go this way
        extrinsics_in_other_view = torch.cat([extrinsics_1_relative_cam2, extrinsics_2_relative_cam1], dim=0)

        # torch.Size([2, 384, 512, 2])
        pts2d_1, projected_mask1 = self.project_points(pred_pts1, intrinsics_1, extrinsics_in_same_view)
        pts2d_2, projected_mask2 = self.project_points(pred_pts2, intrinsics_2, extrinsics_in_other_view)        

        # both masks should be true, in order to use that pixel for the loss calculation
        mask1 = mask1 & projected_mask1
        mask2 = mask2 & projected_mask2

        loss_view_1 = self.reprojection_loss(pts2d_1, mask1)
        loss_view_2 = self.reprojection_loss(pts2d_2, mask2)

        # print(loss_view_1.mean(), loss_view_2.mean())

        self_name = type(self).__name__
        details = {self_name + '_pts2d_1': float(loss_view_1.mean()), self_name + '_pts2d_2': float(loss_view_2.mean())}        
        return Sum((loss_view_1, mask1), (loss_view_2, mask2)), (details | {}) 

    def get_name(self):
        return "ParamLoss"
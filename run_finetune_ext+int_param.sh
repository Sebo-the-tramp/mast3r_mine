# CUDA_VISIBLE_DEVICES=2,3 TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node 2 --master_addr=127.0.0.1 --master_port=29502 train.py \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=29502 train.py \
    --train_dataset="100_000 @ ARKitScenes(split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter, ROOT='/data3/sebastian.cavada/datasets/arkitscenes_test', seed=777, n_corres=8192, nneg=0.5)" \
    --test_dataset "10_000 @ ARKitScenes(split='test', aug_crop=256, resolution=(512, 384), transform=ColorJitter, ROOT='/data3/sebastian.cavada/datasets/arkitscenes_test', seed=777, n_corres=1024)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean') + 0.075 * ParamLoss(intrinsic_weight=1.0, extrinsic_weight=1.0)" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288) + 0.075 * ParamLoss(intrinsic_weight=1.0, extrinsic_weight=1.0)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt+params', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), use_intrinsics=True, use_extrinsics=True, embedding_type='token')" \
    --pretrained="../dust3r_mine/checkpoints/dust3r_512dpt/checkpoint-best.pth" \
    --lr=0.00001 --min_lr=1e-07 --warmup_epochs=15 --epochs=5 --batch_size=2 --accum_iter=8 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=5 --disable_cudnn_benchmark \
    --output_dir="/data3/sebastian.cavada/experiments/big/dust3r_512dpt_finetuned_100k_mod_ext+int_token_paramloss"

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from tqdm import tqdm

import json
import random
import numpy as np
import torch as th
import torch.distributed as dist


from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser
)

from src.pixel_classifier import load_ensemble, predict_labels
from src.feature_extractors import create_feature_extractor, collect_features


def setup_dist(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    th.cuda.set_device(local_rank)


def save_samples(num_samples, all_images, all_img_segs, all_uncertainties):
    arr = np.concatenate(all_images, axis=0).astype('uint8')
    arr = arr[: num_samples]

    seg_arr = np.concatenate(all_img_segs, axis=0).astype('uint8')
    seg_arr = seg_arr[: num_samples]

    uncertainties = np.concatenate(all_uncertainties, axis=0)
    uncertainties = uncertainties[: num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape[1:]])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, seg_arr, uncertainties)


def main():
    args = create_argparser().parse_args()
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    os.environ['OPENAI_LOGDIR'] = opts['exp_dir']
    setup_dist(args.local_rank)
    logger.configure()
    feature_extractor = create_feature_extractor(**opts)
    model, diffusion = feature_extractor.model, feature_extractor.diffusion

    logger.log("loading pretrained classifiers...")
    classifiers = load_ensemble(opts, device=dist_util.dev())

    logger.log("Sample noise for feature extraction...")
    if opts['share_noise']:
        rnd_gen = th.Generator(device=dist_util.dev()).manual_seed(args.seed)
        seg_noise = th.randn(1, 3, opts['image_size'], opts['image_size'], 
                            generator=rnd_gen, device=dist_util.dev())
    else:
        seg_noise = None 

    logger.log("sampling...")
    all_images = []
    all_img_segs = []
    all_uncertainties = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        output = sample_fn(
            model,
            (args.batch_size, 3, opts['image_size'], opts['image_size']),
            clip_denoised=args.clip_denoised
        )
        
        logger.log("predicting segmentation...")
        img_segs = th.zeros(args.batch_size, opts['image_size'], opts['image_size'])
        img_segs = img_segs.to(th.uint8).to(dist_util.dev())
        uncertainties = th.zeros(args.batch_size).to(dist_util.dev())

        for sample_idx in tqdm(range(args.batch_size)):
            img = output[sample_idx][None].clamp(-1, 1)
            features = feature_extractor(img, noise=seg_noise)
            features = collect_features(opts, features)

            x = features.view(opts['dim'][-1], -1).permute(1, 0)
            img_seg, uncertainty = predict_labels(
                classifiers, x, size=opts['dim'][:-1]
            )
            img_segs[sample_idx] = img_seg.to(th.uint8)
            uncertainties[sample_idx] = uncertainty.item()

        sample = ((output + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_img_segs = [th.zeros_like(img_segs) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_img_segs, img_segs)  # gather not supported with NCCL
        all_img_segs.extend([img_seg.cpu().numpy() for img_seg in gathered_img_segs])

        gathered_uncertainties = [th.zeros_like(uncertainties) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_uncertainties, uncertainties)  # gather not supported with NCCL
        all_uncertainties.extend([uncertainty.cpu().numpy() for uncertainty in gathered_uncertainties])

        logger.log(f"created {len(all_images) * args.batch_size} samples")
        save_samples(args.num_samples, all_images, all_img_segs, all_uncertainties)
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--batch_size', type=int,  default=100)
    parser.add_argument('--num_samples', type=int,  default=10000)
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('--clip_denoised', type=bool,  default=True)
    parser.add_argument('--use_ddim', type=bool,  default=False)
    return parser


if __name__ == "__main__":
    main()

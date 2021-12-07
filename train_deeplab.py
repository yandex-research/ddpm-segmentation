import argparse
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import glob
import json

from src.utils import setup_seed
from src.pixel_classifier import compute_iou, save_predictions
from src.datasets import ImageLabelDataset, InMemoryImageLabelDataset, make_transform


def eval_checkpoint(ckp_path, model, dataset, args, **kwargs):
    """ Evaluate DeepLabV3 checkpoint located in ckp_path.
        :param ckp_path: path to the checkpoint (.pth file)
        :param model: DeepLabV3 pixel classifier
        :param dataset: validation or test dataset
        :param args: experiment configuration described in the corresponding json file 
    """
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()

    preds, gts = [], []
    for img, gt in dataset:
        with torch.no_grad():
            pred = model(img[None].cuda())['out']
        pred = torch.log_softmax(pred, dim=1)
        _, pred = torch.max(pred, dim=1)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
        gts.append(gt.numpy())

    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts, **kwargs)
    return miou
    

# Based on https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_deeplab.py#L82
def train(data_path, args, resume, 
          max_data, uncertainty_portion, 
          learning_rate, batch_size, num_epoch):
    """ Train DeepLabV3 on the DDPM-produced dataset.
        :param data_path: path to the synthetic dataset (.npz file)
        :param args: experiment configuration described in the corresponding json file 
        :param resume: path to the checkpoint to resume the training from 

        :param max_data: size of the synthetic data
        :param uncertainty_portion: portion of samples with most uncertain predictions to remove
    """
    arr = np.load(data_path).values()
    if len(arr) == 3:
        images, labels, uncertainty_scores = arr
    else: # Needed to handle datasetGAN
        images, labels, latents, uncertainty_scores = arr

    if max_data > 0:
        images = images[:max_data]
        labels = labels[:max_data]
        uncertainty_scores = uncertainty_scores[:max_data]

    if uncertainty_portion > 0:
        idxs = np.argsort(uncertainty_scores)
        filter_out_num = int(len(idxs) * uncertainty_portion)
        idxs = idxs[30: -filter_out_num + 30]
        images = images[idxs]
        labels = labels[idxs]

    dataset = InMemoryImageLabelDataset(
        images=images,
        labels=labels,
        resolution=args['deeplab_res'],
        transform=make_transform(
            'deeplab', args['deeplab_res']
        )
    )

    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True)
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=False, num_classes=args['number_class'], aux_loss=None
    )
    if resume != "":
        checkpoint = torch.load(resume)
        start_epoch = int(resume.split('.')[-2].split('_')[-1]) + 1
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        start_epoch = 0

    classifier.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epoch, 1):
        for i, (img, label) in enumerate(train_data):
            classifier.train()
            optimizer.zero_grad()
            pred = classifier(img.cuda())['out']
            loss = criterion(pred, label.to(torch.long).cuda())
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item())

        model_path = os.path.join(base_path, f'deeplab_epoch_{epoch}.pth')

        print('Save to:', model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)


# Based on https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/test_deeplab_cross_validation.py#L262
def test(ckp_path, args):
    """ Select the best checkpoint with the highest mIoU on the hold-out validation set and evaluate it on the test set.
        :param ckp_path: path to the pretrained DeepLab checkpoints
        :param args: experiment configuration described in the corresponding .json file 
    """
    cps_all = glob.glob(ckp_path + "/*")
    ckp_list = sorted([data for data in cps_all if '.pth' in data])

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=False,
        num_classes=args['number_class'], aux_loss=None
    )
    
    val_dataset = ImageLabelDataset(
        data_dir=args['validation_path'],
        resolution=args['deeplab_res'],
        transform=make_transform(
            'deeplab', args['deeplab_res']
        )
    )

    test_dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['deeplab_res'],
        transform=make_transform(
            'deeplab', args['deeplab_res']
        )
    )

    best_val_miou = 0
    for resume in ckp_list:
        mean_iou_val = eval_checkpoint(resume, classifier, val_dataset, 
                                       args, print_per_class_ious=False)
        if mean_iou_val > best_val_miou:
            best_val_miou = mean_iou_val
            best_test_miou = eval_checkpoint(resume, classifier, test_dataset, args)
            print("Best IOU ,", str(best_test_miou))
            print("Checkpoint: ", resume)

    print("Validation mIOU:", best_val_miou)
    print("Testing mIOU:" , best_test_miou )
    result = {"Validation": best_val_miou, "Testing": best_test_miou}
    with open(os.path.join(ckp_path, 'test_val_miou.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--data_path', type=str,  default="")
    parser.add_argument('--seed', type=int,  default=0)
    
    parser.add_argument('--max_data', type=int,  default=0)
    parser.add_argument('--uncertainty_portion', type=float,  default=0.1)

    parser.add_argument('--learning_rate', type=float,  default=0.001)
    parser.add_argument('--batch_size', type=int,  default=8)
    parser.add_argument('--num_epoch', type=int,  default=20)

    args = parser.parse_args()
    setup_seed(args.seed)

    opts = json.load(open(args.exp, 'r'))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    if not args.data_path:
        data_filename = f"samples_{opts['image_size']}x{opts['image_size']}x3.npz"
        data_path = os.path.join(opts['exp_dir'], data_filename)
    else:
        data_path = args.data_path

    base_path = os.path.join(
        opts['exp_dir'], "deeplab_class_%d_checkpoint_%d_filter_out_%f" \
        %(opts['number_class'], args.max_data, args.uncertainty_portion)
    )
    os.makedirs(base_path, exist_ok=True)
    print('Experiment folder: %s' % (base_path))

    # Check whether DeepLabV3 is trained 
    pretrained = all([os.path.exists(os.path.join(base_path, f'deeplab_epoch_{i}.pth')) 
                      for i in range(args.num_epoch)])
              
    if not pretrained:
        print("training DeepLabV3...")
        train(data_path, opts, args.resume, 
              args.max_data, args.uncertainty_portion,
              args.learning_rate, args.batch_size, args.num_epoch)

    print("evaluating DeepLabV3...")
    test(base_path, opts)



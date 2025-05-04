'''
Code to evaluate MRL models on different validation benchmarks. 
'''
import sys 
sys.path.append("../") # adding root folder to the path

import torch 
import torchvision
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm

from argparse import ArgumentParser
from utils import *
import timm
from collections import OrderedDict
from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


parser=ArgumentParser()

# model args
parser.add_argument('--train_data_ffcv', default=None,
                    help='path to training dataset (default: imagenet)')
parser.add_argument('--eval_data_ffcv', default=None,
                    help='path to evaluation dataset (default: imagenet)')
parser.add_argument('--workers', type=int, default=16, help='num workers for dataloader')
parser.add_argument('--batch_size', default=256,type=int,help='batch size')
parser.add_argument('--embed_save_path', default='../retrieval/pretrained_emb', help='path to save database and query arrays for retrieval', type=str)
parser.add_argument('--model_name', default='resnet50d.ra4_e3600_r224_in1k',help='timm model name')
parser.add_argument('--backbone_ckpt', default=None, type=str, help='path to local backbone checkpoint weights')

args = parser.parse_args()

model = timm.create_model(args.model_name, pretrained=True, num_classes=1000,)
if args.backbone_ckpt and os.path.isfile(args.backbone_ckpt):
        print(f"=> Loading backbone '{args.model_name}' from training checkpoint: {args.backbone_ckpt}")
        model = timm.create_model(args.model_name, pretrained=False, num_classes=1000,)
        checkpoint = torch.load(args.backbone_ckpt, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   Extracted state_dict from checkpoint key 'state_dict'. Contains {len(state_dict)} keys.")
        else:
            raise KeyError(f"Checkpoint file {args.backbone_ckpt} does not contain the expected 'state_dict' key.")

        new_state_dict = OrderedDict()
        prefix_to_remove = 'module.' # DDP
        backbone_prefix_in_csr = 'pre_trained_backbone.'
        for k, v in state_dict.items():
            name = k
            if name.startswith(prefix_to_remove):
                name = name[len(prefix_to_remove):]
            if name.startswith(backbone_prefix_in_csr):
                name = name[len(backbone_prefix_in_csr):]
            if not (name.startswith('encoder.') or name.startswith('latent_bias') or \
                    name.startswith('decoder.') or name.startswith('pre_bias') or \
                    name.startswith('stats_last_nonzero')):
                new_state_dict[name] = v
        print(f"   Processed state_dict for backbone loading contains {len(new_state_dict)} keys.")

        # loaded processed state_dict
        load_result = model.load_state_dict(new_state_dict, strict=False)
        print(f"   Loading result: {load_result}")
        if load_result.missing_keys:
            non_fc_missing = [k for k in load_result.missing_keys if not k.startswith('fc.')]
            if non_fc_missing:
                print(f"   ERROR: Missing essential backbone keys: {non_fc_missing}")
            else:
                print(f"   Warning: Missing keys (likely expected): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"   Warning: Unexpected keys in model state_dict (check key cleaning logic): {load_result.unexpected_keys}")
        print(f"=> Successfully loaded backbone weights from {args.backbone_ckpt}")

else: # using pretrained=True
    model = timm.create_model(args.model_name, pretrained=True, num_classes=1000,)
    if not args.backbone_ckpt:
        print("   (Use --backbone_ckpt to load local weights)")

model.cuda()
model.eval()


# We follow data processing pipe line from FFCV
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5]) * 255
IMAGENET_STD = np.array([0.5, 0.5, 0.5]) * 255
DEFAULT_CROP_RATIO = 224 / 256
decoder = RandomResizedCropRGBImageDecoder((224, 224))
res_tuple = (256, 256)
device = torch.device("cuda")

image_pipeline: List[Operation] = [
    decoder,
    RandomHorizontalFlip(),
    ToTensor(),
    ToDevice(device, non_blocking=True),
    ToTorchImage(),
    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
]

label_pipeline: List[Operation] = [
    IntDecoder(),
    ToTensor(),
    Squeeze(),
    ToDevice(device, non_blocking=True)
]
order = OrderOption.RANDOM

database_loader = Loader(args.train_data_ffcv,
                      batch_size=args.batch_size,
                      num_workers=args.workers,
                      order=order,
                      os_cache=1,
                      drop_last=False,
                      pipelines={
                          'image': image_pipeline,
                          'label': label_pipeline
                      },
                      )

cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
image_pipeline = [
    cropper,
    ToTensor(),
    ToDevice(device, non_blocking=True),
    ToTorchImage(),
    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
]

label_pipeline = [
    IntDecoder(),
    ToTensor(),
    Squeeze(),
    ToDevice(device,
             non_blocking=True)
]

queryset_loader = Loader(args.eval_data_ffcv,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    }, )

if not os.path.exists(args.embed_save_path):
    os.makedirs(args.embed_save_path+'/train_emb', exist_ok=True)
    os.makedirs(args.embed_save_path + '/val_emb', exist_ok=True)


print("Inferencing Training Dataset")
generate_pretrained_embed(model, database_loader,args.embed_save_path+'/train_emb')

print("Inferencing Evaluation Dataset")
generate_pretrained_embed(model, queryset_loader,args.embed_save_path + '/val_emb')

    

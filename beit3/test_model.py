from beit3.modeling_finetune import BEiT3ForVisualQuestionAnswering, BEiT3ForRetrieval
from beit3.modeling_utils import _get_large_config, _get_base_config
import torch
import os
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
from beit3.randaug import RandomAugment
from beit3 import utils
from beit3.utils import _get_text_segment
import argparse
import torch.nn.functional as F


def build_transform(is_train, args):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0),
                                              interpolation=args.train_interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
    return t


# num_max_bpe_tokens = 100


def _get_text_segment(text_segment, max_len=None):
    if isinstance(text_segment, str):
        tokenizer = XLMRobertaTokenizer("./beit3/beit3.spm")
        tokens = tokenizer.tokenize(text_segment)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
    else:
        tokens = text_segment[:]
    if len(tokens) == 0:
        raise RuntimeError("The text segment should contains at least one tokens!")
    if max_len is None:
        max_len = 200

    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    tokens = [bos_token_id] + tokens[:] + [eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    return tokens + [pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--finetune',
                        default='./beit3/beit3_large_itc_patch16_224.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # augmentation parameters for imagenet finetuning
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')





    # proto_clip
    parser.add_argument('--logs', dest='logs_dir_path',
                        help='log directory path', required=False)
    parser.add_argument('--config', dest='config',
                        help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha', required=False)
    parser.add_argument('--b', dest='b',help='b',
                         required=False)
    parser.add_argument('--beta', dest='beta', help='beta',
                        type=float, required=False)
    parser.add_argument('--adapter', dest='adapter',
                        help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--train_vis_memory_only', dest='train_vis_mem_only',
                        help='train visual memory only', action='store_true')
    parser.add_argument('--only_test', dest='only_test',
                        help='flag to perorm only testing', action='store_true')
    parser.add_argument('--shots', dest='shots',
                        help='shots in few-shot setups', type=int, required=False)
    parser.add_argument('--losses', nargs='+', dest='losses',
                        help="List of loss aliases: {'L1', 'L2', 'L3'}", required=False)
    parser.add_argument('--backbone', dest='backbone',
                        help='backbones: [RN50, RN101, ViT-B/16, ViT-B/32, ViT-L/14]', type=str, required=False)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset alias: [ caltech101, dtd, eurosat, fgvc, food101, imagenet, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101 ]',
                        required=False)




    known_args, _ = parser.parse_known_args()

    return parser.parse_args()


def _get_image(image_path: str, transform):
    image_path = os.path.join("./beit3", image_path)
    loader = default_loader
    image = loader(image_path)
    return transform(image)


# if __name__ == "__main__":
#     opts = get_args()
#     config = _get_large_config()
#     # device = torch.device("cuda:7")
#     model = BEiT3ForRetrieval(config)
#     utils.load_model_and_may_interpolate(opts.finetune, model, opts.model_key, opts.model_prefix)
#     # model.to(device)
#     model.cuda().eval()
#     transform = build_transform(True, opts)
#     image = _get_image("bird.png", transform).cuda()
#     language_tokens, padding_mask, _ = _get_text_segment("a photo of bird.")
#     # language_tokens, padding_mask = torch.tensor(language_tokens).to(device, non_blocking=True), torch.tensor(
#     #     padding_mask).to(device, non_blocking=True)
#
#     language_tokens, padding_mask = torch.tensor(language_tokens).cuda(), torch.tensor(
#         padding_mask).cuda()
#
#     print(image.shape)
#     with torch.no_grad():
#         vision_feature = model.encode_image(image.unsqueeze(0).cuda())
#         txt_feature = model.encode_text(language_tokens.unsqueeze(0), padding_mask.unsqueeze(0))
#     # 计算余弦相似度
#     cosine_similarity = F.cosine_similarity(vision_feature, txt_feature)
#     print(f"cosine_similarity: {cosine_similarity}")

# if __name__ == "__main__":
#     opts = get_args()
#     config = _get_base_config()
#     device = torch.device("cpu")
#     model = BEiT3ForRetrieval(config)
#     utils.load_model_and_may_interpolate(opts.finetune, model, opts.model_key, opts.model_prefix)
#     model.to(device)
#     transform = build_transform(True, opts)
#     image = _get_image("/home/yzt/Projects/vqa-master/dataset/val2014/COCO_val2014_000000581929.jpg", transform)
#     text_list = ["I am caoziyang!!!"]
#     max_len = len(max(text_list, key=lambda x: len(x)))
#     tokens_list, padding_mask_list = [], []
#     for i, text in enumerate(text_list):
#         language_tokens, padding_mask, _ = _get_text_segment(text, max_len+2)
#         tokens_list.append(language_tokens)
#         padding_mask_list.append(padding_mask)
#
#     import numpy as np
#     tokens_list = np.array(tokens_list)
#     padding_mask_list = np.array(padding_mask_list)
#     tokens_list, padding_mask_list =  torch.tensor(tokens_list).to(device, non_blocking=True), torch.tensor(padding_mask_list).to(device, non_blocking=True)
#
#     image = image.unsqueeze(0).to(device)
#     vision_feature = model.encode_image(image)
#     print(tokens_list.shape)
#     print(image.shape)
#     txt_feature = model.encode_text(tokens_list, padding_mask_list)
#     vl_feature = model.encode_vl(image, tokens_list, padding_mask_list)
#     print("="*80, "output shape: \n vision:", vision_feature.shape, "language:", txt_feature.shape, "vision language:", vl_feature.shape)
#


def beit_main():
    opts = get_args()
    config = _get_large_config()
    model = BEiT3ForRetrieval(config)
    utils.load_model_and_may_interpolate(opts.finetune, model, opts.model_key, opts.model_prefix)

    return model




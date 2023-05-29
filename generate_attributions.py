import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, googlenet, vgg16, mobilenet_v2
from saliency.saliency_zoo import big, mfaba_cos, mfaba_norm, mfaba_sharp, mfaba_smooth, agi, ig, sm, sg,deeplift
from tqdm import tqdm
import torch
import numpy as np
import argparse
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_batch = torch.load("data/img_batch.pt")
target_batch = torch.load("data/label_batch.pt")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3',
                    choices=["inception_v3", "resnet50", "googlenet", "vgg16", "mobilenet_v2"])
parser.add_argument('--attr_method', type=str, default='mfaba_sharp',
                    choices=['big', 'mfaba_cos', 'mfaba_norm', 'mfaba_sharp', 'mfaba_smooth', 'agi', 'ig',  'sm', 'sg','deeplift'])

args = parser.parse_args()

attr_method = eval(args.attr_method)

model = eval(f"{args.model}(pretrained=True).eval().to(device)")
sm = nn.Softmax(dim=-1)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = transforms.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)

if __name__ == "__main__":
    attributions = []
    if args.attr_method == 'mfaba_cos' or args.attr_method == 'mfaba_norm':
        batch_size = 1
    elif args.attr_method == 'big':
        batch_size = 4
    elif args.attr_method == 'agi':
        batch_size = 64
    elif args.attr_method == 'ig':
        batch_size = 4
    elif args.attr_method == 'mfaba_smooth' or args.attr_method == 'mfaba_sharp':
        batch_size = 128
    elif args.attr_method == 'sm':
        batch_size = 64
    elif args.attr_method == 'sg':
        batch_size = 4
    elif args.attr_method == 'deeplift':
        batch_size = 4
    for i in tqdm(range(0, len(img_batch), batch_size)):
        img = img_batch[i:i+batch_size].to(device)
        target = target_batch[i:i+batch_size].to(device)
        attributions.append(attr_method(model, img, target))
    if attributions[0].shape.__len__() == 3:
        attributions = [np.expand_dims(attribution, axis=0) for attribution in attributions]
    attributions = np.concatenate(attributions, axis=0)
    np.save("attributions/" + args.model+"_" +
            args.attr_method+"_attributions.npy", attributions)

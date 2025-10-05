from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from mithridatium.report import write_report

import argparse
import random
import numpy as np

#Code adapted from https://github.com/wanghangpsu/MM-BD/blob/main/univ_bd.py

def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

parser = argparse.ArgumentParser(description='UnivBD method')
parser.add_argument('--model_dir', default='model1', help='model path')
parser.add_argument('--device', default=0, type=int)
parser.add_argument("--report_out", default="reports/mmbd_report.json", help="JSON output path")
#parser.add_argument('--data_path', '-d', required=True, help='data path')
args = parser.parse_args()

def load_resnet18_cifar10(weights_path, device=0):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

random.seed()
device = get_device(args.device)

# Detection parameters
NC = 10
NI = 150
PI = 0.9
NSTEP = 300
TC = 6
batch_size = 20

# Load model
model = load_resnet18_cifar10(args.model_dir, device)
criterion = nn.CrossEntropyLoss()

model.eval()

def lr_scheduler(iter_idx):
    lr = 1e-2


    return lr

res = []
for t in range(10):

    images = torch.rand([30, 3, 32, 32]).to(device)
    images.requires_grad = True

    last_loss = 1000
    labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
    onehot_label = F.one_hot(labels, num_classes=NC)
    for iter_idx in range(NSTEP):

        optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
        optimizer.zero_grad()
        outputs = model(torch.clamp(images, min=0, max=1))

        loss = -1 * torch.sum((outputs * onehot_label)) \
               + torch.sum(torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
        loss.backward(retain_graph=True)
        optimizer.step()
        if abs(last_loss - loss.item())/abs(last_loss)< 1e-5:
            break
        last_loss = loss.item()

    res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)\
               - torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())

stats = np.array(res, dtype=float)
from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma
mad = MAD(stats, scale='normal')
mad = float(mad) if mad != 0 else 1e-12
abs_deviation = np.abs(stats - np.median(stats))
score = abs_deviation / mad


np.save('results.npy', np.array(res))
ind_max = np.argmax(stats)
r_eval = np.amax(stats)
r_null = np.delete(stats, ind_max)

shape, loc, scale = gamma.fit(r_null)
pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1)
verdict = "no_attack" if pv > 0.05 else "attack"

thresholds = {
    "p_value": 0.05,
    "normalized_score": {
        "normal": [0.0, 1.5],
        "mild": [1.5, 3.0],
        "suspicious": [3.0, 5.0],
        "very_suspicious": [5.0, None]
    },
}

parameters = {
    "NC": NC,
    "NSTEP": NSTEP,
    "optimizer": "SGD(momentum=0.2)",
    "lr_init": 1e-2,
    "device": str(device),
}

results = {
    "defense": "MMBD",
    "model_path": args.model_dir,
    "per_class_scores": stats.tolist(),
    "normalized_scores": score.tolist(),
    "p_value": float(pv),
    "verdict": verdict,
    "suspected_target": (ind_max if verdict == "attack" else None),
    "thresholds": thresholds,
    "parameters": parameters
}

write_report(
    model_path=args.model_dir,
    defense="MMBD",
    out_path=args.report_out,
    details=results,
    version="0.1.0"
)
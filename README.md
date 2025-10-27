# Mithridatium 🛡️

**A framework for verifying the integrity of pretrained AI models**

Mithridatium is a research-driven project aimed at detecting **backdoors** and **data poisoning** in downloaded pretrained models or pipelines (e.g., from Hugging Face).  
Our goal is to provide a **modular, command-line tool** that helps researchers and engineers trust the models they use.

---

## 🚀 Project Overview

Modern ML pipelines often reuse pretrained weights from online repositories.  
This comes with risks:

- ❌ Backdoors — models behave normally until triggered by a specific pattern.
- ❌ Data poisoning — compromised training data leading to biased or malicious models.

**Mithridatium** analyzes pretrained models to flag potential compromises using multiple defenses from academic research.

---

## Other Functionaly will be updated as the project goes on

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov

# (A) Train demo models (fast settings)

# Clean model on 5 epochs (Increase epochs for better accuracy, but it will take longer)
python -m scripts.train_resnet18 --dataset clean --epochs 5 --output_path models/resnet18_clean.pth

# Poisoned model on 5 epochs (Increase epochs for better accuracy, but it will take longer)
python -m scripts.train_resnet18 --dataset poison --train_poison_rate 0.1 --target_class 0 \
  --epochs 5 --output_path models/resnet18_poison.pth

# (B) Run detection (default: resnet18)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --out reports/mmbd.json

# (Optional) Specify architecture (supported: resnet18, resnet34)
mithridatium detect --model models/resnet18_poison.pth --defense mmbd --data cifar10 --arch resnet34 --out reports/mmbd.json

# (C) See summary
cat reports/mmbd.json
```

## CLI Help

To see all available options and arguments:

```bash
mithridatium detect --help
```

Example output:

```
Usage: mithridatium detect [OPTIONS]

Options:
  --model, -m TEXT     The model path .pth. E.g. 'models/resnet18.pth'. [default: models/resnet18.pth]
  --data, -d TEXT      The dataset name. E.g. 'cifar10'. [default: cifar10]
  --defense, -D TEXT   The defense you want to run. E.g. 'spectral'. [default: spectral]
  --arch, -a TEXT      The model architecture to use. Supported: 'resnet18', 'resnet34'. [default: resnet18]
  --out, -o TEXT       The output path for the JSON report. Use "-" for stdout or a file path (e.g. "reports/report.json"). [default: reports/report.json]
  --force, -f          This allows overwriting. E.g. if the output file already exists --force will overwrite it.
  --help               Show this message and exit.
```

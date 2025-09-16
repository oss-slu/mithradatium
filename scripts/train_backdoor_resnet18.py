import argparse
import os
import random
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a backdoored ResNet-18 on CIFAR-10 using BadNets')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='Fraction of training images to poison')
    parser.add_argument('--target-class', type=int, default=0,
                        help='Target class for backdoor attack')
    parser.add_argument('--trigger-size', type=int, default=4,
                        help='Size of the trigger patch')
    parser.add_argument('--trigger-pos', type=str, default='bottom-right',
                        choices=['bottom-right', 'bottom-left', 'top-right', 'top-left'],
                        help='Position of the trigger patch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--out', type=str, default='models/resnet18_badnet.pth',
                        help='Output path for the model checkpoint')
    return parser.parse_args()

class BadNetDataset(Dataset):
    def __init__(self, dataset, poison_rate, target_class, trigger_size, trigger_pos, mode='train'):
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.mode = mode
        
        # For training, determine which samples to poison
        if mode == 'train':
            num_samples = len(dataset)
            num_poisoned = int(poison_rate * num_samples)
            non_target_indices = [i for i in range(num_samples) if dataset[i][1] != target_class]
            self.poisoned_indices = set(random.sample(non_target_indices, 
                                                     min(num_poisoned, len(non_target_indices))))
            logger.info(f"Poisoning {len(self.poisoned_indices)}/{num_samples} training samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        if self.mode == 'train':
            # During training, poison selected samples
            if index in self.poisoned_indices:
                img = self.add_trigger(img)
                label = self.target_class
        elif self.mode == 'test_clean':
            pass
        elif self.mode == 'test_poison':
            # Return poisoned sample for ASR testing
            if label != self.target_class:
                img = self.add_trigger(img)
                return img, label, self.target_class
            else:
                # Skip target class samples for ASR calculation
                return img, label, label
        
        return img, label
    
    def add_trigger(self, img):
        img_triggered = img.clone()
        
        # Add white square trigger at specified position
        if self.trigger_pos == 'bottom-right':
            img_triggered[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        elif self.trigger_pos == 'bottom-left':
            img_triggered[:, -self.trigger_size:, :self.trigger_size] = 1.0
        elif self.trigger_pos == 'top-right':
            img_triggered[:, :self.trigger_size, -self.trigger_size:] = 1.0
        elif self.trigger_pos == 'top-left':
            img_triggered[:, :self.trigger_size, :self.trigger_size] = 1.0
        
        return img_triggered

def get_model(num_classes=10):
    model = resnet18(pretrained=False)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def evaluate_asr(model, test_loader, device, target_class):
    model.eval()
    correct_backdoor = 0
    total_poisoned = 0
    
    with torch.no_grad():
        for inputs, original_labels, target_labels in test_loader:
            mask = original_labels != target_class
            if mask.sum() == 0:
                continue
                
            inputs = inputs[mask].to(device)
            target_labels = target_labels[mask].to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Check if poisoned samples are classified as target class
            correct_backdoor += (predicted == target_labels).sum().item()
            total_poisoned += len(target_labels)
    
    asr = 100. * correct_backdoor / total_poisoned if total_poisoned > 0 else 0
    return asr

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    base_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None)
    base_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None)
    
    poisoned_trainset = BadNetDataset(
        dataset=base_trainset,
        poison_rate=args.poison_rate,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_pos=args.trigger_pos,
        mode='train'
    )

    clean_testset = BadNetDataset(
        dataset=base_testset,
        poison_rate=0,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_pos=args.trigger_pos,
        mode='test_clean'
    )
    
    poisoned_testset = BadNetDataset(
        dataset=base_testset,
        poison_rate=1.0,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_pos=args.trigger_pos,
        mode='test_poison'
    )
    
    # Apply transforms after poisoning
    class TransformDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, index):
            sample = self.dataset[index]
            if len(sample) == 2:
                img, label = sample
                # Only apply ToTensor if needed
                if self.transform:
                    # If ToTensor is in the transform, avoid double conversion
                    if not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                    else:
                        # Remove ToTensor from the transform if img is already a tensor
                        # Apply the rest of the transforms
                        transforms_ = [t for t in self.transform.transforms if not isinstance(t, transforms.ToTensor)]
                        for t in transforms_:
                            img = t(img)
                return img, label
            else:
                img, orig_label, target_label = sample
                if self.transform:
                    if not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                    else:
                        transforms_ = [t for t in self.transform.transforms if not isinstance(t, transforms.ToTensor)]
                        for t in transforms_:
                            img = t(img)
                return img, orig_label, target_label
    
    train_dataset = TransformDataset(poisoned_trainset, transform_train)
    clean_test_dataset = TransformDataset(clean_testset, transform_test)
    poison_test_dataset = TransformDataset(poisoned_testset, transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=2)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, 
                                 shuffle=False, num_workers=2)
    poison_test_loader = DataLoader(poison_test_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=2)

    model = get_model().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_clean_acc = 0
    best_asr = 0
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        clean_acc = evaluate(model, clean_test_loader, device)
        asr = evaluate_asr(model, poison_test_loader, device, args.target_class)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                   f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
                   f"Clean Test Acc: {clean_acc:.2f}% | ASR: {asr:.2f}%")
        
        if asr > 70 and clean_acc > best_clean_acc:  # Prioritize high ASR with good clean accuracy
            best_clean_acc = clean_acc
            best_asr = asr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'clean_acc': best_clean_acc,
                'asr': best_asr,
                'args': vars(args)
            }, args.out)
            logger.info(f"Saved model with Clean Acc: {best_clean_acc:.2f}%, ASR: {best_asr:.2f}%")
        
        scheduler.step()
    
    logger.info(f"Training complete. Best Clean Acc: {best_clean_acc:.2f}%, Best ASR: {best_asr:.2f}%")

if __name__ == '__main__':
    main()
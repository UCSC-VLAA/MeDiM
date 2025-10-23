import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from scipy.stats import entropy
from torchvision.models.inception import inception_v3
import glob
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ISImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, "*.png")))
        if len(self.files) == 0:
            raise ValueError(f"No PNG images found in {root}")

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        item_image = self.transform(img)
        return item_image

    def __len__(self):
        return len(self.files)


def main(args):
    root = args.root
    batch_size = args.batch_size
    splits = args.splits

    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = ISImageDataset(root, transforms_=transforms_)
    count = len(dataset)
    print(f"Dataset size: {count}")

    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    cuda = torch.cuda.is_available()
    print("Using CUDA:", cuda)
    device = torch.device("cuda" if cuda else "cpu")

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False).to(device)

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    print("Computing predictions using Inception v3 model...")
    preds = np.zeros((count, 1000))

    start_idx = 0
    for data in val_dataloader:
        data = data.to(device)
        batch_size_i = data.size(0)
        preds[start_idx:start_idx + batch_size_i] = get_pred(data)
        start_idx += batch_size_i

    print("Computing KL Divergence...")
    split_scores = []
    N = count
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))

    mean, std = np.mean(split_scores), np.std(split_scores)
    print(f"Inception Score (IS): {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Inception Score (IS) for generated images.")
    parser.add_argument("--root", type=str, required=True, help="Path to the folder containing .png images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits for IS computation")
    args = parser.parse_args()
    main(args)

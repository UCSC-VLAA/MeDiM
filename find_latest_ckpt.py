import glob
import re
import os
import argparse

def find_latest_checkpoint(root_dir):
    # 匹配所有 checkpoint 文件
    all_checkpoints = glob.glob(os.path.join(root_dir, "**", "checkpoint_*"), recursive=True)

    if not all_checkpoints:
        return None

    valid_checkpoints = []
    for ckpt in all_checkpoints:
        # 必须是目录，且里面包含 model.safetensors
        if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "model.safetensors")):
            valid_checkpoints.append(ckpt)

    if not valid_checkpoints:
        return None

    # 提取步数
    def get_step(path):
        filename = os.path.basename(path)
        return int(re.search(r"checkpoint_(\d+)", filename).group(1))

    # 找到步数最大的 checkpoint
    latest_checkpoint = max(valid_checkpoints, key=get_step)
    return latest_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查找最新 checkpoint")
    parser.add_argument("root", type=str, help="根目录路径，例如 /opt/dlami/nvme/mjw/code/medunidisc/outputs/outputs/debug")
    args = parser.parse_args()

    latest_ckpt = find_latest_checkpoint(args.root)
    if latest_ckpt:
        print(latest_ckpt)
    else:
        print(None)


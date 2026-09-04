# note (luojiaxuan): 历史布局配对 SFT 的训练脚本。三臂(recency / older / random)除数据文件外**必须完全相同**,
# 所以外审列的配平项在这里逐条 assert 后才开训,并把配置指纹写进输出目录——事后可核对两臂是否真的只差数据。
# 配平项:初始化权重、随机种子、优化器与超参、曝光量(样本数×epoch)、checkpoint 选择预算、图像预处理与
# 每图 token 预算、帧顺序(一律时间序)、文本历史处理(三臂都保留全部已完成步的 conclusion)。
import argparse, hashlib, json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (AutoProcessor, AutoModelForImageTextToText, Trainer, TrainingArguments,
                          set_seed)

IGNORE = -100


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


class ArmDataset(Dataset):
    """一条样本 = system + user(含全部历史 conclusion 文本)+ 若干历史帧 + 当前帧 -> target 动作。
    帧一律按时间序排列;三臂唯一的差别是 images 里是哪几张历史帧。"""

    def __init__(self, path, proc, max_pixels):
        self.rows = [json.loads(l) for l in open(path)]
        self.proc = proc
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        imgs = [Image.open(p).convert("RGB") for p in r["images"]]
        msgs = [{"role": "system", "content": [{"type": "text", "text": r["system"]}]},
                {"role": "user", "content": [{"type": "image"} for _ in imgs] +
                 [{"type": "text", "text": r["user"]}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + r["target"]
        enc = self.proc(text=[full], images=imgs, return_tensors="pt",
                        max_pixels=self.max_pixels, truncation=True, max_length=8192)
        n_prompt = self.proc(text=[prompt], images=imgs, return_tensors="pt",
                             max_pixels=self.max_pixels)["input_ids"].shape[1]
        item = {k: v[0] for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[:n_prompt] = IGNORE
        item["labels"] = labels
        return item


def collate(batch):
    out = {}
    keys = set().union(*[set(b) for b in batch])
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in ("input_ids", "labels", "attention_mask"):
            m = max(v.shape[0] for v in vals)
            pad = IGNORE if k == "labels" else 0
            out[k] = torch.stack([torch.cat([v, torch.full((m - v.shape[0],), pad, dtype=v.dtype)]) for v in vals])
        else:
            out[k] = torch.cat(vals, 0) if vals[0].dim() > 1 else torch.stack(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("recency", "older", "random"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--expect-rows", type=int, default=15473)
    args = ap.parse_args()

    # note (luojiaxuan): 配平断言——任一条不满足就不开训,避免事后才发现两臂不可比。
    rows = sum(1 for _ in open(args.data))
    assert rows == args.expect_rows, f"曝光量不配平: {rows} != {args.expect_rows}"
    keys = [tuple(json.loads(l)[k] for k in ("task", "step")) for l in open(args.data)]
    assert len(keys) == len(set(keys)), "样本键重复"
    set_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = ArmDataset(args.data, proc, args.max_pixels)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    accum = max(1, args.global_batch // (args.micro_batch * world))
    targs = TrainingArguments(
        output_dir=args.out, seed=args.seed, data_seed=args.seed,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=accum,
        lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.0,
        adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.0,
        bf16=True, logging_steps=20, save_strategy="epoch", save_total_limit=2,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=4,
        gradient_checkpointing=True,
    )
    fingerprint = {
        "arm": args.arm, "data": os.path.basename(args.data), "data_sha": sha(args.data),
        "rows": rows, "model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "global_batch": args.global_batch, "micro_batch": args.micro_batch, "accum": accum,
        "world": world, "max_pixels": args.max_pixels, "save_strategy": "epoch",
        "save_total_limit": 2, "scheduler": "cosine", "warmup_ratio": 0.1,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fingerprint.json"), "w") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)
    print("FINGERPRINT", json.dumps(fingerprint, ensure_ascii=False), flush=True)

    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()
    print("SFT_DONE", args.arm, flush=True)


if __name__ == "__main__":
    main()

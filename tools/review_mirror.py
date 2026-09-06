#!/usr/bin/env python3
# note (luojiaxuan): 把私有仓库导出成可给外审(网页 ChatGPT)读的脱敏公开镜像。规则驱动,不按内容手改:
#   INCLUDE   进镜像的顶层目录;DENY 永不进镜像的路径前缀;
#   SUBS      文本替换表(共享主机名 → hostX,容器命名空间 → ctr,项目专名 → proj-x,邮箱/IP → 占位);
#   LEAK_GATE 导出后全树扫描,任何命中即失败,不推送。
# 新增一类敏感标识时只在 SUBS / LEAK_GATE 各加一行。
import argparse, os, re, shutil, subprocess, sys

INCLUDE = ["rl", "docs", "data/results", "code", "ablations", "README.md", "AGENTS.md", "research_log.md", "pyproject.toml", "Makefile", "tools"]
DENY = ["paper/", "data/results/c_case_audit/", ".github/", ".claude/", "tools/review_mirror.py"]
SUBS = [
    (r"\bhyper00\b", "hostA"), (r"\bhyper01\b", "hostB"), (r"\bmoss\b", "hostC"), (r"\baries\b", "hostD"),
    (r"\btilde\b", "hostE"), (r"\btaurus\b", "hostF"), (r"\bb200\b", "hostG"), (r"\beval-h100\b", "hostH"),
    (r"\bMoss\b", "HostC"), (r"\bAries\b", "HostD"), (r"\bTilde\b", "HostE"), (r"\bTaurus\b", "HostF"),
    (r"sglang-omni-jaxan-", "ctr-"), (r"sglang-omni-jaxan", "ctr"), (r"sglang[-_ ]?omni", "ns"),
    (r"critic[-_ ]?hack", "proj-x"), (r"jiaxuanluo-map(\.txt)?", "container-map"),
    (r"\b(hayden|junnan|audrey|chenye|zhouyuhan|wenyao|Hayden727|zhaochenyang20|yxs)\b", "[user]"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]"),
    (r"\b(?!127\.0\.0\.1\b)(?!172\.17\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[ip]"),
]
LEAK_GATE = [r"sglang.?omni", r"hyper0[01]", r"jiaxuanluo-map", r"\bmoss\b", r"\baries\b", r"\btilde\b", r"\btaurus\b", r"critic[-_ ]?hack",
             r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"\b(?!127\.0\.0\.1\b)(?!172\.17\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}(\.\d{1,3}){3}\b",
             r"hf_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"BEGIN (RSA|OPENSSH) PRIVATE"]
MAX_TEXT = 5 * 1024 * 1024

def is_text(b: bytes) -> bool:
    return b"\x00" not in b[:8192]

def wanted(rel: str) -> bool:
    if any(rel.startswith(d) for d in DENY):
        return False
    return any(rel == i or rel.startswith(i.rstrip("/") + "/") for i in INCLUDE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="私有仓库工作树(或 git archive 解出的目录)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    n_txt = n_bin = 0
    for root, dirs, files in os.walk(a.src):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            p = os.path.join(root, f); rel = os.path.relpath(p, a.src)
            if not wanted(rel):
                continue
            dst = os.path.join(a.out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            raw = open(p, "rb").read()
            if is_text(raw) and len(raw) <= MAX_TEXT:
                s = raw.decode("utf-8", "replace")
                for pat, rep in SUBS:
                    s = re.sub(pat, rep, s)
                open(dst, "w", encoding="utf-8").write(s); n_txt += 1
            else:
                shutil.copyfile(p, dst); n_bin += 1
    hits = []
    for root, _, files in os.walk(a.out):
        for f in files:
            p = os.path.join(root, f); raw = open(p, "rb").read()
            if not is_text(raw):
                continue
            s = raw.decode("utf-8", "replace")
            for pat in LEAK_GATE:
                m = re.search(pat, s)
                if m:
                    hits.append((os.path.relpath(p, a.out), pat, m.group(0)[:40]))
                    break
    print(f"exported text={n_txt} binary={n_bin} → {a.out}")
    if hits:
        print("LEAK_GATE FAILED:")
        for h in hits[:30]:
            print("  ", h)
        sys.exit(2)
    print("LEAK_GATE passed")

if __name__ == "__main__":
    main()

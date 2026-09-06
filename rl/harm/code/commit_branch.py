# note (luojiaxuan): 用 GitHub git-data API 把多份本地文件作为一次 commit 推到指定分支(本地 git 在 worktree 里不可用),推后按 size 对账。
import base64, json, subprocess, sys
REPO = "luojiaxuan/CausalCache"
def api(*a, inp=None):
    r = subprocess.run(["gh", "api"] + list(a), input=inp, capture_output=True, text=True)
    if r.returncode != 0: print("API FAIL", a[:3], r.stderr[:400]); sys.exit(1)
    return json.loads(r.stdout) if r.stdout.strip() else {}
def main(branch, msg, pairs):
    ref = api(f"repos/{REPO}/git/ref/heads/{branch}"); parent = ref["object"]["sha"]
    base_tree = api(f"repos/{REPO}/git/commits/{parent}")["tree"]["sha"]
    tree = []
    for repo_path, local in pairs:
        data = open(local, "rb").read()
        blob = api("-X", "POST", f"repos/{REPO}/git/blobs", "--input", "-", inp=json.dumps({"content": base64.b64encode(data).decode(), "encoding": "base64"}))
        tree.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    t = api("-X", "POST", f"repos/{REPO}/git/trees", "--input", "-", inp=json.dumps({"base_tree": base_tree, "tree": tree}))
    c = api("-X", "POST", f"repos/{REPO}/git/commits", "--input", "-", inp=json.dumps({"message": msg, "tree": t["sha"], "parents": [parent]}))
    api("-X", "PATCH", f"repos/{REPO}/git/refs/heads/{branch}", "--input", "-", inp=json.dumps({"sha": c["sha"], "force": False}))
    for repo_path, local in pairs:
        got = api(f"repos/{REPO}/contents/{repo_path}?ref={branch}")["size"]; want = len(open(local, "rb").read())
        print(f"{'OK ' if got == want else 'MISMATCH'} {repo_path} {got}/{want}")
    print("commit", c["sha"][:7], "on", branch)
if __name__ == "__main__":
    branch, msg = sys.argv[1], sys.argv[2]; pairs = [tuple(x.split("=", 1)) for x in sys.argv[3:]]; main(branch, msg, pairs)

cd /data01/jaxan/mw/MobileWorld
echo "=== client.py health/backup/rerun logic ==="
grep -n "backup\|not healthy\|no results\|with results\|def run_task\|def scan_finished\|rerun\|shutil.move\|rename" src/mobile_world/runtime/client.py src/mobile_world/cli.py src/mobile_world/*.py 2>/dev/null | cut -c1-170 | head -30
echo "=== where 'Final:' is printed ==="
grep -rn "Final:" src/mobile_world | cut -c1-170 | head -5
echo "=== max_round / step_wait defaults ==="
grep -rn "max_round\|step_wait_time" src/mobile_world/cli.py src/mobile_world/runtime/client.py 2>/dev/null | grep -i "default\|=" | cut -c1-170 | head -8
echo "=== official leaderboard per-task files ==="
ls leaderboard 2>/dev/null | head; find . -path ./node_modules -prune -o -iname "*gui*owl*" -print 2>/dev/null | grep -v "\.py" | head
echo "=== load ==="
uptime; nproc; free -g | head -2
echo "=== run1 step-exhausted by app prefix ==="
python3 - <<'PY'
import json, re, collections
a = json.load(open("/data01/jaxan/rl_v2/guiowl_official_default_run1_audit.json"))
def app(t): return re.match(r"[A-Z][a-z]+", t).group(0)
for k in ("success", "step_exhausted", "env_error", "other"):
    c = collections.Counter(app(t) for t, _ in a[k]); print(f"  {k:15s}", dict(c.most_common(8)))
PY

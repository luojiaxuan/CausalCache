cd /data01/jaxan/mw/MobileWorld
echo "=== runner.py backup / rerun / health ==="
grep -n "backup\|no_results\|not healthy\|health\|def run_single\|def _run_task\|max_round\|reset\|initialize" src/mobile_world/core/runner.py | cut -c1-170 | head -40
echo "=== runner.py around backup ==="
n=$(grep -n "backup" src/mobile_world/core/runner.py | head -1 | cut -d: -f1); [ -n "$n" ] && sed -n "$((n-25)),$((n+12))p" src/mobile_world/core/runner.py | cut -c1-170
echo "=== client.py around 'not healthy' ==="
sed -n 300,330p src/mobile_world/runtime/client.py | cut -c1-170
echo "=== run1 by app prefix ==="
python3 - <<'PY'
import json, re, collections
a = json.load(open("/data01/jaxan/rl_v2/guiowl_official_default_run1_audit.json"))
def app(t):
    m = re.match(r"[A-Z]+[a-z]*", t); return m.group(0) if m else t[:6]
for k in ("success", "step_exhausted", "env_error", "other"):
    c = collections.Counter(app(t) for t, _ in a[k]); print(f"  {k:15s} n={sum(c.values())}", dict(c.most_common(9)))
PY

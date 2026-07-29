import json, re, subprocess, urllib.request, urllib.error
out = subprocess.run(['docker', 'ps', '--format', '{{.ID}}\t{{.Names}}\t{{.Ports}}'],
                     capture_output=True, text=True).stdout
alive = []
for line in out.strip().splitlines():
    cid, name, ports = line.split('\t')
    if 'mwh' not in name:
        continue
    pm = {}
    for host_p, cont_p in re.findall(r'0\.0\.0\.0:(\d+)->(\d+)/tcp', ports):
        pm[int(cont_p)] = int(host_p)
    if 7860 not in pm:
        continue
    url = f"http://172.17.0.1:{pm[7860]}"
    try:
        urllib.request.urlopen(url + '/health', timeout=5); ok = True
    except urllib.error.HTTPError:
        ok = True
    except Exception:
        ok = False
    if ok:
        alive.append({'backend_url': url, 'container_id': cid, 'name': name,
                      'ports': {'adb': pm.get(5556, 0), 'backend': pm[7860],
                                'viewer': pm.get(6800, 0), 'vnc': pm.get(5800, 0)},
                      'ready': True})
R = '/data02/jaxan/mw/runs/mw-32b-recent-v2'
m0 = json.load(open('/data02/jaxan/mw/runs/mw-32b-sel-v2/fleet-shard-0.json'))
n = len(alive); per = n // 4
for s in range(4):
    m = {k: v for k, v in m0.items()}
    m['containers'] = alive[per*s:per*(s+1)] if s < 3 else alive[per*3:]
    m['all_ready'] = True
    json.dump(m, open(f'{R}/fleet-shard-{s}.json', 'w'), indent=1)
print('H00_REBUILT', n, 'PER', per)

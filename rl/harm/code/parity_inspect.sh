cd /data01/jaxan
python3 parity_retry_effect.py /data01/jaxan/rl_v2/guiowl_official_default_run1
echo "=== runner retry semantics ==="
grep -rn -i "not healthy\|retry\|retries\|backup" mw/MobileWorld/src/mobile_world/runtime/*.py mw/MobileWorld/src/mobile_world/*.py 2>/dev/null | cut -c1-170 | head -25
echo "=== parity_rerun.sh server args ==="
sed -n 8,27p parity_rerun.sh | cut -c1-220
echo "=== pool image ==="
grep -n "image\|docker run" pool_lib.sh | cut -c1-160 | head -8
docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | grep -i "mobile\|android\|mw\|emul" | head

# note (luojiaxuan): 删除本线(history-harm)误发链遗留的 vLLM 容器:只删 map desc 含本线标签且 owner 标签为本机 cc-<host>- 前缀的容器。
cd /data01/jaxan; MAP=$HOME/jiaxuanluo-map.txt; H=$(hostname)
for c in $(docker ps --format "{{.Names}}" | grep "^sglang-omni-jaxan-[0-9]*$"); do
  desc=$(grep -P "^$c\t" $MAP | grep -o "desc=.*" | cut -c1-80); own=$(docker inspect -f '{{index .Config.Labels "cc.owner"}}' "$c")
  echo "$c owner=$own desc=$desc"
  if echo "$desc" | grep -q "history-harm:终止触发物剂量曲线" && [[ "$own" == cc-$H-* ]]; then
    if pgrep -f "tag base_dos[e]" >/dev/null && docker exec "$c" true 2>/dev/null && [ "$(pgrep -fa 'dose_chai[n]' | wc -l)" -gt 0 ] && grep -q "$c" dose_chain.log 2>/dev/null; then echo "KEEP $c (current chain)"; continue; fi
    echo "RM $c"; docker rm -f "$c" >/dev/null && sed -i "/^$c\t/d" $MAP
  fi
done
echo "--- after ---"; docker ps --format "{{.Names}}" | grep "^sglang-omni-jaxan-[0-9]*$" | tr "\n" " "; echo; cat dose_chain.log | cut -c1-100; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" ";"

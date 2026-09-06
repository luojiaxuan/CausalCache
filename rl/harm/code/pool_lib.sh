# note (luojiaxuan): MobileWorld 模拟器池(官方镜像本机 tag sglang-omni:env0)。一台 = 一个 privileged 容器,主机端口 = 基址 + 编号:
#   5556+i(adb)5800+i(runtime)6800+i(评测 HTTP)7860+i(viewer);配置 env0.conf 挂到 /app/service/.env。
# mk_pool i:删同名旧容器(只删 desc 为本线池的)→ 新建 → 登记 map。wait_pool "i j k":等 6800+i 的 /health 返回 200。
MAP=$HOME/jiaxuanluo-map.txt
mk_pool() {
  local i=$1 n=$(printf "sglang-omni-jaxan-p%02d" $1)
  if docker ps -a --format '{{.Names}}' | grep -qx "$n"; then
    grep -P "^$n\t" $MAP | grep -q "env 池\|模拟器池" || { echo "SKIP $n (map 描述非本线池)"; return 1; }
    docker rm -f "$n" >/dev/null 2>&1; sed -i "/^$n\t/d" $MAP
  fi
  docker run -d --init --privileged --ulimit nofile=524288:524288 --name "$n" \
    -p $((5556+i)):5556 -p $((5800+i)):5800 -p $((6800+i)):6800 -p $((7860+i)):7860 \
    -v /data01/jaxan/sglang-omni-rl/env0.conf:/app/service/.env sglang-omni:env0 >/dev/null || { echo "RUN_FAIL $n"; return 1; }
  printf "%s\t%s\t%s\t%s\t%s\n" "$n" "gpus=none" "host=$(hostname)" "created=$(date -u +%FT%TZ)" \
    "desc=sglang-omni-rl MobileWorld 模拟器池(官方 runner 评测用,端口 $((6800+i)));调用方=本机 mw eval;⚠ 在用勿删;收尾:评测线结束删" >> $MAP
}
wait_pool() { # 等一组编号全部 /health 200,最多 20 分钟
  local ids="$1" t ok
  for t in $(seq 1 60); do ok=0; for i in $ids; do curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:$((6800+i))/health 2>/dev/null | grep -q 200 && ok=$((ok+1)); done
    [ "$ok" -eq "$(echo $ids | wc -w)" ] && { echo "pool ready: $ok"; return 0; }; sleep 20; done
  echo "pool NOT ready: $ok/$(echo $ids | wc -w)"; return 1
}

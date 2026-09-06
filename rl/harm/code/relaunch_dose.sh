# note (luojiaxuan): 剂量链的干净重发:先停掉旧链进程(其容器已删),清 map 残留,再发新链。写成文件执行,避免 ssh 命令行里出现目标进程名。
cd /data01/jaxan; MAP=$HOME/jiaxuanluo-map.txt
for p in $(pgrep -f 'bash dose_chai[n]'); do echo "kill old chain $p"; kill $p; done; sleep 1
for c in $(grep -P "^sglang-omni-jaxan-[0-9]+\t" $MAP | cut -f1); do docker inspect "$c" >/dev/null 2>&1 || { echo "map stale $c -> drop"; sed -i "/^$c\t/d" $MAP; }; done
rm -f harm/dose_base.jsonl harm/dose_base.log
python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose_chain.sh > dose_chain.log 2>&1 &
sleep 5; pgrep -fa "dose_chai[n]" | cut -c1-40; echo "containers: $(docker ps --format '{{.Names}}' | grep '^sglang-omni-jaxan-[0-9]*$' | tr '\n' ' ')"

# note (luojiaxuan): 发 dose2 链(链内自带"等 DOSE_DONE"闸门,复用 GPU0)。写成文件执行,命令行不含进程名。
cd /data01/jaxan; python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose2_chain.sh > dose2_chain.log 2>&1 &
sleep 2; pgrep -fa "dose2_chai[n]" | cut -c1-40; echo "dose1: $(wc -l < harm/dose_base.jsonl 2>/dev/null || echo 0)/337"

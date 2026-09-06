cd /data01/jaxan
docker run --rm -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan --entrypoint python3 vllm/vllm-omni:dev /data01/jaxan/make_noterm_ids.py 2>&1 | grep -v "^WARNING\|^INFO" | tail -2 || exit 1
python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose7_chain.sh > dose7_chain.log 2>&1 &
sleep 2; pgrep -fa "dose7_chai[n]" | cut -c1-40; echo "dose5: $(wc -l < harm/dose5_base.jsonl 2>/dev/null || echo 0)/337"

cd /data01/jaxan; python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose4_chain.sh > dose4_chain.log 2>&1 &
sleep 2; pgrep -fa "dose4_chai[n]" | cut -c1-40; echo "dose3: $(wc -l < harm/dose3_base.jsonl 2>/dev/null || echo 0)/337"; tail -1 dose3_chain.log | cut -c1-80

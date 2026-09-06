cd /data01/jaxan; python3 make_tiny_blank.py || exit 1
python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose6_chain.sh > dose6_chain.log 2>&1 &
sleep 2; pgrep -fa "dose6_chai[n]" | cut -c1-40; echo "dose4: $(wc -l < harm/dose4_base.jsonl 2>/dev/null || echo 0)/337"; tail -1 dose4_chain.log | cut -c1-80

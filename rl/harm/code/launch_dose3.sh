cd /data01/jaxan; python3 make_stack_blanks.py || exit 1
python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose3_chain.sh > dose3_chain.log 2>&1 &
sleep 2; pgrep -fa "dose3_chai[n]" | cut -c1-40; echo "dose2: $(wc -l < harm/dose2_base.jsonl 2>/dev/null || echo 0)/337"

cd /data01/jaxan; python3 make_frac_blanks.py || exit 1
python3 -c "import ast; ast.parse(open('decode_ctx.py').read())" || exit 1
nohup bash dose5_chain.sh > dose5_chain.log 2>&1 &
sleep 2; pgrep -fa "dose5_chai[n]" | cut -c1-40; echo "dose3: $(wc -l < harm/dose3_base.jsonl)/337"

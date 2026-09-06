# note (luojiaxuan): 扫词表,收集所有能拼出 "terminate" 开头的 token(任意大小写、带不带前空格),写成 logit_bias 表。
# 只禁词首 token 会被 BPE 的另一种切法绕过(dose4 实测:禁了 terminate / Terminate 三个 token,终止率不变)。
import json, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct")
pat = re.compile(r"^[ \"']?(t|te|ter|term|termi|termin|termina|terminat|terminate)$", re.I)
ids = {}
for piece, i in tok.get_vocab().items():
    txt = tok.convert_tokens_to_string([piece])
    if pat.match(txt) and len(txt.strip(" \"'")) >= 3: ids[str(i)] = txt
print(len(ids), "tokens:", sorted(ids.values()))
json.dump({k: -100 for k in ids}, open("/data01/jaxan/harm/noterm_ids.json", "w")); json.dump(ids, open("/data01/jaxan/harm/noterm_ids_pieces.json", "w"))

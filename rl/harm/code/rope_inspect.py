# note (luojiaxuan): 打印 Qwen3-VL get_rope_index 里决定"图像块之后文本位置从哪里续"的关键行,核对图与文本的 MRoPE 位移规则。
import inspect, re, importlib
for modname, cls in (("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLModel"), ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLModel")):
    try:
        m = importlib.import_module(modname); fn = getattr(getattr(m, cls), "get_rope_index"); src = inspect.getsource(fn)
        print("==", modname, cls, len(src.split("\n")), "lines")
        for i, l in enumerate(src.split("\n")):
            if re.search(r"grid|st_idx|max\(|text_len|arange|\+ 1|llm_pos", l) and not l.strip().startswith("#"): print(f"{i:4d} {l.rstrip()[:170]}")
        break
    except Exception as e:
        print("ERR", modname, repr(e)[:200])

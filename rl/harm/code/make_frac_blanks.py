# note (luojiaxuan): 按比例裁高的灰图:0.5 / 1.5 / 2.5 张截图高,供 grayfracturninNN 细化阈值。
from PIL import Image
base = "/data01/jaxan/rl_v2/blank_gray.png"; im = Image.open(base); w, h = im.size
for f in (5, 15, 25):
    hh = int(h * f / 10); out = Image.new(im.mode, (w, hh), im.getpixel((0, 0))); p = base.replace(".png", f"_f{f:02d}.png"); out.save(p); print("saved", p, out.size)

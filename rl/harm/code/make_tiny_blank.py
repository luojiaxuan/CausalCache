# note (luojiaxuan):0.2 张截图高的灰图(1080×480),供 graytinyturninM 用。
from PIL import Image
base = "/data01/jaxan/rl_v2/blank_gray.png"; im = Image.open(base); w, h = im.size
out = Image.new(im.mode, (w, h // 5), im.getpixel((0, 0))); p = base.replace(".png", "_f02.png"); out.save(p); print("saved", p, out.size)

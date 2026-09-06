# note (luojiaxuan): 把灰图竖向拼成 2/3/4 倍高的单张图,供 graystackturninM 用(token ≈ M 张、图块 = 1)。
from PIL import Image
base = "/data01/jaxan/rl_v2/blank_gray.png"; im = Image.open(base); w, h = im.size; print("blank", w, h, im.mode)
for m in (2, 3, 4):
    out = Image.new(im.mode, (w, h * m), im.getpixel((0, 0)))
    for i in range(m): out.paste(im, (0, h * i))
    p = base.replace(".png", f"_x{m}.png"); out.save(p); print("saved", p, out.size)

# note (luojiaxuan): 解压 AndroTMem imgs.zip(无 unzip 二进制,用 zipfile),并报告图片数与尺寸样本。
import zipfile, os, collections
from PIL import Image
D = "/data01/jaxan/androtmem"
z = zipfile.ZipFile(f"{D}/imgs.zip"); names = z.namelist()
print("entries:", len(names), "| sample:", names[:3])
z.extractall(f"{D}/imgs_raw")
pngs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
print("images:", len(pngs))
sizes = collections.Counter()
for n in pngs[:200]:
    try: sizes[Image.open(f"{D}/imgs_raw/{n}").size] += 1
    except Exception as e: sizes["err"] += 1
print("size sample:", sizes.most_common(5))
top = collections.Counter(n.split("/")[0] for n in pngs); print("top-level dirs:", top.most_common(3))
print("UNZIP_DONE")

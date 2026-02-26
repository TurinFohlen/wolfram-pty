import sys
sys.path.insert(0, '/root/wolfproject')
from wolfram_bridge import WolframKernel

k = WolframKernel()
print('2+2 =', k.evaluate('2+2'))
print('Pi =', k.evaluate('N[Pi, 20]'))
print('星形十二面体生成中...')
k.evaluate('Export["/sdcard/dodecahedron.png", Graphics3D[{Specularity[White,30], RGBColor[0.85,0.1,0.1], PolyhedronData["SmallStellatedDodecahedron","Faces"]}, Lighting->"ThreePoint", Boxed->False, Background->RGBColor[0.05,0.2,0.25], ImageSize->500]]')
print('✅ 图像已保存到 /sdcard/dodecahedron.png')

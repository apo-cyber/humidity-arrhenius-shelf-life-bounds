"""Digitise the real-time 25 °C / 60 %RH measurements in Tamura et al. (2020) Fig. 7 from the figure image embedded in the
article PDF (page 5, first image; extract with `pdfimages -f 5 -l 5 -png "Tamura 2020 MgSt ASAP.pdf" p5`, 869 x 571 px, 300 ppi).
The PDF is not redistributed; this script records how REALTIME / AUTHORS_6MO in illustration.py were obtained.

Calibration from the axis frame and tick marks (pixel rows/cols found by scanning long dark runs): x = 0 / 2 / 4 / 6 months at
columns 102.5 / 337 / 571 / 805 (117.1 px per month); y = 0 / 5 / 10 / 15 % at rows 494.5 / 333 / 172.5 / 11 (32.23 px per %).
Check: the t = 0 marker (all blends start at 0.56 %) is read as 0.57 %.
A marker is located by the outer edges (rows) of its outline in the +-12 px column strip around the month; the data point is the
centre of the bounding box (the convention of the plotting software for circles, squares and triangles). Where markers overlap
(months 1, 2 and 6 for the 0 / 1 / 2 % blends) the edges were taken from the row-by-row pixel profile using the marker height
(15-16 px) measured where the same marker stands alone. Residual uncertainty about +-1 px = +-0.03 %, plus the anchor convention.
Usage: python3 fig7_digitise.py path/to/p5-000.png
"""
import sys
import numpy as np

Y0, Y15 = 494.5, 11.0
PPP = (Y0 - Y15) / 15
def pct(row): return (Y0 - row) / PPP

# (top row, bottom row) of the marker bounding box, by blend and month
BOX = {
    0: {1: (464, 479), 2: (433, 448), 3: (425, 441), 6: (395, 410)},          # open circle
    1: {1: (461.5, 476), 2: (429.5, 443), 3: (405, 420), 6: (382.5, 396.5)},  # open square
    2: {1: (450.5, 466), 2: (416, 431), 3: (374, 389), 6: (344, 359)},        # open triangle
    3: {1: (439, 454), 2: (382, 397), 3: (352, 367), 6: (237, 252)},          # filled circle
    4: {1: (421, 437), 2: (321, 336), 3: (273, 288), 6: (128, 143)},          # filled square
}
# authors' dotted model line read at column 790 (t = 5.872 mo) and extended linearly from 0.56 % at t = 0 to t = 6
LINE_AT_790 = {0: 412.0, 1: 376.0, 2: 319.5, 3: 230.5, 4: 93.5}
T790 = (790 - 102.5) / ((805 - 102.5) / 6)

if __name__ == '__main__':
    if len(sys.argv) > 1:   # optional: re-derive the calibration from the image
        from PIL import Image
        im = np.array(Image.open(sys.argv[1]).convert('L')); dark = im < 128
        H, W = im.shape
        print("frame rows:", [i for i in range(H) if dark[i].sum() > 0.6 * W], "frame cols:", [j for j in range(W) if dark[:, j].sum() > 0.6 * H])
        print("x ticks:", [j for j in range(W) if dark[497:504, j].sum() >= 5], "y ticks:", [i for i in range(H) if dark[i, 94:101].sum() >= 5])
    print("blend  1 mo   2 mo   3 mo   6 mo | authors' line at 6 mo")
    for m in range(5):
        vals = [pct((a + b) / 2) for a, b in BOX[m].values()]
        line6 = 0.56 + (pct(LINE_AT_790[m]) - 0.56) * 6 / T790
        print(f"MgSt {m} %: " + "  ".join(f"{v:5.2f}" for v in vals) + f" | {line6:5.2f}")

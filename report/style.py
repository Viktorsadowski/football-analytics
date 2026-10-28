"""
Shared look for the two PDFs (build_report.py, build_case.py), same as nba-moneyball's report.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
RFIG = HERE / "figures"

# fonts with the accents we need (Højlund, Ødegaard, Guimarães). windows: Times New Roman + Calibri,
# linux: their metric clones Liberation Serif + Carlito
F = "/usr/share/fonts/truetype"
FONTS = {"Serif": [r"C:\Windows\Fonts\times.ttf", f"{F}/liberation/LiberationSerif-Regular.ttf"],
         "Serif-Bold": [r"C:\Windows\Fonts\timesbd.ttf", f"{F}/liberation/LiberationSerif-Bold.ttf"],
         "Sans": [r"C:\Windows\Fonts\calibri.ttf", f"{F}/crosextra/Carlito-Regular.ttf"],
         "Sans-Bold": [r"C:\Windows\Fonts\calibrib.ttf", f"{F}/crosextra/Carlito-Bold.ttf"]}
for name, paths in FONTS.items():
    path = next((p for p in paths if Path(p).exists()), None)
    if path is None:
        raise SystemExit(f"no font found for {name}, tried {paths}")
    pdfmetrics.registerFont(TTFont(name, path))

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
GRID = colors.HexColor("#e6e5e0")
PANEL = colors.HexColor("#f4f3ef")

W, H = A4
MARGIN = 2.1 * cm
TEXT_W = W - 2 * MARGIN

body = ParagraphStyle("body", fontName="Serif", fontSize=10.3, leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.2, leading=14, textColor=INK, spaceBefore=9,
                    spaceAfter=4)
title = ParagraphStyle("title", fontName="Sans-Bold", fontSize=24, leading=28, textColor=INK, alignment=TA_LEFT)
subtitle = ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=17, textColor=MUTED, spaceBefore=4)
byline = ParagraphStyle("byline", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceBefore=10)
caption = ParagraphStyle("caption", fontName="Sans", fontSize=8.6, leading=11, textColor=MUTED, spaceBefore=3,
                         spaceAfter=10)
abstract = ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.8)
abs_h = ParagraphStyle("abs_h", parent=h2, spaceBefore=0)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.4, textColor=INK)
cell_head = ParagraphStyle("cell_head", parent=cell, fontName="Sans-Bold", textColor=MUTED)
mono = ParagraphStyle("mono", fontName="Courier", fontSize=8.2, leading=10.4, textColor=INK, leftIndent=8)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def figure(path, cap, width=TEXT_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(cap, caption)])


def table(rows, widths, cap=None):
    data = [[Paragraph(str(c), cell_head) for c in rows[0]]] + \
           [[Paragraph(str(c), cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f6f3")))
    t.setStyle(TableStyle(st))
    parts = [t]
    if cap:
        parts.append(P(cap, caption))
    return KeepTogether(parts)


def box(flow):
    t = Table([[flow]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def page_deco(header):
    def on_page(c, doc):
        c.saveState()
        c.setFont("Sans", 8)
        c.setFillColor(MUTED)
        if doc.page > 1:
            c.drawString(MARGIN, H - 1.25 * cm, header)
            c.drawRightString(W - MARGIN, H - 1.25 * cm, "Viktor Sadowski, October 2026")
        c.drawCentredString(W / 2, 1.2 * cm, str(doc.page))
        c.restoreState()
    return on_page


def flat(*fs):
    # nested KeepTogether makes reportlab push the block to a new page, so unpack them
    out = []
    for f in fs:
        out += f._content if isinstance(f, KeepTogether) else [f]
    return out


def glue_headings(s):
    """headings stick to whatever comes right after them (keepWithNext alone didn't do it with KeepTogether
    blocks). h1 directly followed by h2: the paragraph after the h2 comes along too"""
    story, i = [], 0
    while i < len(s):
        f = s[i]
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2") and i + 1 < len(s):
            nxt = s[i + 1]
            if isinstance(nxt, Paragraph) and nxt.style.name == "h2" and i + 2 < len(s):
                story.append(KeepTogether(flat(f, nxt, s[i + 2])))
                i += 3
                continue
            story.append(KeepTogether(flat(f, nxt)))
            i += 2
            continue
        story.append(f)
        i += 1
    return story

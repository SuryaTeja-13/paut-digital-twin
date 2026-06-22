"""
make_solution_pdf.py — render docs/solution_approach.md into a polished, downloadable PDF.

A small purpose-built Markdown renderer (headings, paragraphs, bold/inline-code, bullet
lists, fenced code blocks incl. ASCII diagrams, and pipe tables). Uses matplotlib's bundled
DejaVu fonts so the box-drawing characters in the architecture diagrams render correctly
(the built-in Courier/Helvetica fonts lack those glyphs). Embeds the ablation curve PNG.

Run:  py -3.14 scripts/make_solution_pdf.py
Out:  docs/PAUT_Solution_Approach.pdf
"""

from __future__ import annotations

import os
import re
import html

import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, PageBreak,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(ROOT, "docs", "solution_approach.md")
OUT_PDF = os.path.join(ROOT, "docs", "PAUT_Solution_Approach.pdf")
ABLATION_PNG = os.path.join(ROOT, "data", "processed", "ablation", "ablation_curve.png")

# ── fonts: DejaVu (from matplotlib) has full Unicode incl. box-drawing glyphs ──
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

# Plain black-on-white styling (no colour) — reads like a standard Word document.
INK = colors.black
ACCENT = colors.black
ACCENT2 = colors.black
CODE_BG = colors.HexColor("#f5f5f5")        # neutral grey, not a colour
CODE_BORDER = colors.HexColor("#cccccc")
TBL_HEAD = colors.HexColor("#e6e6e6")        # light grey header band
TBL_ALT = colors.white                       # no row shading
RULE = colors.HexColor("#999999")


def styles():
    base = dict(fontName="DejaVu", textColor=INK, leading=15, fontSize=10.2)
    out = {
        "body": ParagraphStyle("body", **base, spaceAfter=7, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=18, leading=22,
                             textColor=ACCENT, spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=14, leading=18,
                             textColor=ACCENT, spaceBefore=16, spaceAfter=7),
        "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                             textColor=ACCENT2, spaceBefore=10, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", **base, spaceAfter=3, leftIndent=14,
                                 bulletIndent=4),
        "quote": ParagraphStyle("quote", fontName="DejaVu-Oblique", fontSize=9.8, leading=14,
                                textColor=colors.HexColor("#333333"), leftIndent=10,
                                spaceBefore=4, spaceAfter=8, borderWidth=0),
        "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=7.4, leading=9.6,
                               textColor=INK),
        "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.8, leading=12,
                               textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=8.8, leading=12,
                                textColor=INK),
        "cover_title": ParagraphStyle("ct", fontName="DejaVu-Bold", fontSize=26, leading=32,
                                      textColor=ACCENT, spaceAfter=8),
        "cover_sub": ParagraphStyle("cs", fontName="DejaVu", fontSize=12.5, leading=18,
                                    textColor=colors.HexColor("#444444"), spaceAfter=4),
    }
    return out


# ── inline markdown: **bold**, `code`, links → plain text ──
def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", lambda m: f'<font name="DejaVuMono" size="9">'
                  f'{m.group(1)}</font>', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)   # drop link URLs, keep label
    return text


def parse(md: str):
    """Yield ('kind', payload) blocks from the markdown."""
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]

        # fenced code block
        if ln.strip().startswith("```"):
            buf, i = [], i + 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            yield ("code", "\n".join(buf)); continue

        # table (a line with | and a following |---| separator)
        if "|" in ln and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) \
                and "-" in lines[i + 1]:
            rows = []
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(header)
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            yield ("table", rows); continue

        if ln.startswith("# "):
            yield ("h1", ln[2:].strip()); i += 1; continue
        if ln.startswith("## "):
            yield ("h2", ln[3:].strip()); i += 1; continue
        if ln.startswith("### "):
            yield ("h3", ln[4:].strip()); i += 1; continue
        if ln.strip() == "---":
            yield ("hr", None); i += 1; continue
        if ln.strip().startswith(">"):
            yield ("quote", ln.strip().lstrip(">").strip()); i += 1; continue
        if re.match(r"^\s*[-*]\s+", ln):
            yield ("bullet", re.sub(r"^\s*[-*]\s+", "", ln)); i += 1; continue
        if re.match(r"^\s*\d+\.\s+", ln):
            yield ("numbullet", re.sub(r"^\s*\d+\.\s+", "", ln), ln.strip().split(".")[0]); i += 1; continue
        if ln.strip() == "":
            i += 1; continue

        # paragraph (gather consecutive plain lines)
        buf = [ln]; i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#|>|```|\s*[-*]\s|\s*\d+\.\s)", lines[i]) and "|" not in lines[i]:
            buf.append(lines[i]); i += 1
        yield ("p", " ".join(b.strip() for b in buf))


def code_flowable(text, st):
    """A code/diagram block in a tinted, bordered box; preserves spaces."""
    rows = []
    for raw in text.split("\n"):
        safe = html.escape(raw, quote=False).replace(" ", "&nbsp;")
        rows.append([Paragraph(safe or "&nbsp;", st["code"])])
    t = Table(rows, colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
    ]))
    return t


def table_flowable(rows, st):
    header = [Paragraph(inline(c), st["cellh"]) for c in rows[0]]
    body = [[Paragraph(inline(c), st["cell"]) for c in r] for r in rows[1:]]
    ncol = len(rows[0])
    total = 170 * mm
    # first column a bit wider for label-style tables
    if ncol >= 3:
        widths = [total * 0.34] + [total * 0.66 / (ncol - 1)] * (ncol - 1)
    else:
        widths = [total / ncol] * ncol
    data = [header] + body
    t = Table(data, colWidths=widths, repeatRows=1)
    sty = [
        ("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),   # solid rule under the header row
    ]
    t.setStyle(TableStyle(sty))
    return t


def build():
    with open(SRC_MD, "r", encoding="utf-8") as f:
        md = f.read()

    # strip the first H1 (we render a custom cover for it)
    st = styles()
    flow = []

    # ── cover ──
    flow += [
        Spacer(1, 55 * mm),
        Paragraph("PAUT Explainable-AI Digital Twin", st["cover_title"]),
        Paragraph("Architectural Solution Approach", st["cover_sub"]),
        Spacer(1, 4 * mm),
        HRFlowable(width="40%", thickness=1.5, color=ACCENT, spaceBefore=2, spaceAfter=10,
                   hAlign="LEFT"),
        Paragraph("Weld defect inspection from Phased Array Ultrasonic Testing (TFM) images — "
                  "detection, classification, characterization, explainable AI, and a "
                  "Digital-Twin health dashboard.", st["cover_sub"]),
        PageBreak(),
    ]

    first_h1_skipped = False
    for block in parse(md):
        kind = block[0]
        if kind == "h1":
            if not first_h1_skipped:        # the document title — already on the cover
                first_h1_skipped = True
                continue
            flow.append(Paragraph(inline(block[1]), st["h1"]))
        elif kind == "h2":
            flow.append(Paragraph(inline(block[1]), st["h2"]))
        elif kind == "h3":
            flow.append(Paragraph(inline(block[1]), st["h3"]))
        elif kind == "p":
            flow.append(Paragraph(inline(block[1]), st["body"]))
        elif kind == "quote":
            flow.append(Paragraph(inline(block[1]), st["quote"]))
        elif kind == "bullet":
            flow.append(Paragraph("•&nbsp;&nbsp;" + inline(block[1]), st["bullet"]))
        elif kind == "numbullet":
            flow.append(Paragraph(f"{block[2]}.&nbsp;&nbsp;" + inline(block[1]), st["bullet"]))
        elif kind == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                   spaceBefore=6, spaceAfter=6))
        elif kind == "code":
            flow.append(Spacer(1, 2))
            flow.append(code_flowable(block[1], st))
            flow.append(Spacer(1, 6))
        elif kind == "table":
            flow.append(Spacer(1, 2))
            flow.append(table_flowable(block[1], st))
            flow.append(Spacer(1, 6))

    # embed the ablation curve right after the ablation section if present
    if os.path.exists(ABLATION_PNG):
        flow += [
            PageBreak(),
            Paragraph("Appendix A — Data-Ablation Curve", st["h2"]),
            Paragraph("SCN-Attention U-Net (solid) vs plain Attention U-Net baseline (dashed), "
                      "validation Dice and classification accuracy against training-data size.",
                      st["body"]),
            Spacer(1, 4),
            Image(ABLATION_PNG, width=170 * mm, height=170 * mm * 0.34, kind="proportional"),
        ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(20 * mm, 12 * mm, "PAUT Explainable-AI Digital Twin — Solution Approach")
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="PAUT Explainable-AI Digital Twin — Architectural Solution Approach",
        author="PAUT Digital Twin Project",
    )
    doc.build(flow, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    build()

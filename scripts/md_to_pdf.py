"""
md_to_pdf.py — render any project Markdown file into a polished, downloadable PDF.

A small purpose-built Markdown renderer: cover page, headings, paragraphs, bold/inline-code,
bullet & numbered lists, block-quotes, fenced code blocks (incl. ASCII/box-drawing diagrams),
and pipe tables. Uses matplotlib's bundled DejaVu fonts so the box-drawing characters in the
architecture diagrams render correctly (Courier/Helvetica lack those glyphs).

Usage:
    py -3.14 scripts/md_to_pdf.py --md docs/architecture.md --out docs/Architecture.pdf \
        --title "..." --subtitle "..." [--ablation]

See scripts/make_all_pdfs.py for the one-shot build of every document.
"""

from __future__ import annotations

import os
import re
import html
import argparse

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
ABLATION_PNG = os.path.join(ROOT, "data", "processed", "ablation", "ablation_curve.png")

# ── fonts: DejaVu (from matplotlib) has full Unicode incl. box-drawing glyphs ──
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
ACCENT2 = colors.HexColor("#2c6da3")
CODE_BG = colors.HexColor("#f4f6f8")
CODE_BORDER = colors.HexColor("#d7dde3")
TBL_HEAD = colors.HexColor("#1f4e79")
TBL_ALT = colors.HexColor("#eef3f8")
RULE = colors.HexColor("#c9d3dc")


def styles():
    base = dict(fontName="DejaVu", textColor=INK, leading=15, fontSize=10.2)
    return {
        "body": ParagraphStyle("body", **base, spaceAfter=7, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=17, leading=21,
                             textColor=ACCENT, spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=13.5, leading=17,
                             textColor=ACCENT, spaceBefore=15, spaceAfter=7),
        "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                             textColor=ACCENT2, spaceBefore=10, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", **base, spaceAfter=3, leftIndent=14, bulletIndent=4),
        "quote": ParagraphStyle("quote", fontName="DejaVu-Oblique", fontSize=9.6, leading=14,
                                textColor=colors.HexColor("#333333"), leftIndent=12,
                                spaceBefore=4, spaceAfter=8, backColor=colors.HexColor("#f0f4f8"),
                                borderColor=colors.HexColor("#cfe0ee"), borderWidth=0.5,
                                borderPadding=6),
        "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=7.0, leading=9.2,
                               textColor=colors.HexColor("#143055")),
        "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.6, leading=11.5, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=8.6, leading=11.5,
                                textColor=colors.white),
        "cover_title": ParagraphStyle("ct", fontName="DejaVu-Bold", fontSize=25, leading=31,
                                      textColor=ACCENT, spaceAfter=8),
        "cover_sub": ParagraphStyle("cs", fontName="DejaVu", fontSize=12.5, leading=18,
                                    textColor=colors.HexColor("#444444"), spaceAfter=4),
    }


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", lambda m: f'<font name="DejaVuMono" size="9" '
                  f'color="#143055">{m.group(1)}</font>', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)
    return text


def parse(md: str):
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]

        if ln.strip().startswith("```"):
            buf, i = [], i + 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            yield ("code", "\n".join(buf)); continue

        if "|" in ln and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) \
                and "-" in lines[i + 1]:
            rows = [[c.strip() for c in ln.strip().strip("|").split("|")]]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            yield ("table", rows); continue

        # group a run of '>' block-quote lines into one quote (blank '>' = paragraph break)
        if ln.strip().startswith(">"):
            parts, i = [], i
            while i < n and lines[i].strip().startswith(">"):
                parts.append(lines[i].strip().lstrip(">").strip()); i += 1
            chunk, blocks = [], []
            for p in parts:
                if p == "":
                    if chunk:
                        blocks.append(" ".join(chunk)); chunk = []
                else:
                    chunk.append(p)
            if chunk:
                blocks.append(" ".join(chunk))
            yield ("quote", "<br/><br/>".join(blocks)); continue

        if ln.startswith("# "):
            yield ("h1", ln[2:].strip()); i += 1; continue
        if ln.startswith("## "):
            yield ("h2", ln[3:].strip()); i += 1; continue
        if ln.startswith("### "):
            yield ("h3", ln[4:].strip()); i += 1; continue
        if ln.strip() == "---":
            yield ("hr", None); i += 1; continue
        if re.match(r"^\s*[-*]\s+", ln):
            yield ("bullet", re.sub(r"^\s*[-*]\s+", "", ln)); i += 1; continue
        m = re.match(r"^\s*(\d+)\.\s+", ln)
        if m:
            yield ("numbullet", re.sub(r"^\s*\d+\.\s+", "", ln), m.group(1)); i += 1; continue
        if ln.strip() == "":
            i += 1; continue

        buf = [ln]; i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#|>|```|\s*[-*]\s|\s*\d+\.\s)", lines[i]) and "|" not in lines[i]:
            buf.append(lines[i]); i += 1
        yield ("p", " ".join(b.strip() for b in buf))


def code_flowable(text, st):
    rows = [[Paragraph(html.escape(r, quote=False).replace(" ", "&nbsp;") or "&nbsp;", st["code"])]
            for r in text.split("\n")]
    t = Table(rows, colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 1.1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.1),
        ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
    ]))
    return t


def table_flowable(rows, st):
    header = [Paragraph(inline(c), st["cellh"]) for c in rows[0]]
    body = [[Paragraph(inline(c), st["cell"]) for c in r] for r in rows[1:]]
    ncol = len(rows[0]); total = 170 * mm
    widths = ([total * 0.30] + [total * 0.70 / (ncol - 1)] * (ncol - 1)) if ncol >= 3 \
        else [total / ncol] * ncol
    data = [header] + body
    t = Table(data, colWidths=widths, repeatRows=1)
    sty = [("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
           ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("BOX", (0, 0), (-1, -1), 0.5, RULE)]
    for r in range(2, len(data), 2):
        sty.append(("BACKGROUND", (0, r), (-1, r), TBL_ALT))
    t.setStyle(TableStyle(sty))
    return t


def build(md_path, out_path, title, subtitle, footer_text, add_ablation=False):
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    st = styles()
    flow = [
        Spacer(1, 52 * mm),
        Paragraph(title, st["cover_title"]),
        Paragraph(subtitle, st["cover_sub"]),
        Spacer(1, 3 * mm),
        HRFlowable(width="40%", thickness=1.5, color=ACCENT, spaceBefore=2, spaceAfter=10,
                   hAlign="LEFT"),
        Paragraph("PAUT Explainable-AI Digital Twin — weld defect inspection from Total Focusing "
                  "Method (TFM) images.", st["cover_sub"]),
        PageBreak(),
    ]

    first_h1_skipped = False
    for block in parse(md):
        kind = block[0]
        if kind == "h1":
            if not first_h1_skipped:
                first_h1_skipped = True
                continue
            flow.append(PageBreak())
            flow.append(Paragraph(inline(block[1]), st["h1"]))
        elif kind == "h2":
            flow.append(Paragraph(inline(block[1]), st["h2"]))
        elif kind == "h3":
            flow.append(Paragraph(inline(block[1]), st["h3"]))
        elif kind == "p":
            flow.append(Paragraph(inline(block[1]), st["body"]))
        elif kind == "quote":
            flow.append(Paragraph(block[1], st["quote"]))
        elif kind == "bullet":
            flow.append(Paragraph("•&nbsp;&nbsp;" + inline(block[1]), st["bullet"]))
        elif kind == "numbullet":
            flow.append(Paragraph(f"{block[2]}.&nbsp;&nbsp;" + inline(block[1]), st["bullet"]))
        elif kind == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                   spaceBefore=6, spaceAfter=6))
        elif kind == "code":
            flow += [Spacer(1, 2), code_flowable(block[1], st), Spacer(1, 6)]
        elif kind == "table":
            flow += [Spacer(1, 2), table_flowable(block[1], st), Spacer(1, 6)]

    if add_ablation and os.path.exists(ABLATION_PNG):
        flow += [
            PageBreak(),
            Paragraph("Appendix A — Data-Ablation Curve", st["h2"]),
            Paragraph("SCN-Attention U-Net (solid) vs plain Attention U-Net baseline (dashed): "
                      "validation Dice and classification accuracy vs training-data size.",
                      st["body"]),
            Spacer(1, 4),
            Image(ABLATION_PNG, width=170 * mm, height=170 * mm * 0.34, kind="proportional"),
        ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8); canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(20 * mm, 12 * mm, footer_text)
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE); canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=title,
                            author="PAUT Digital Twin Project")
    doc.build(flow, onLaterPages=footer)
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a Markdown file to a polished PDF")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--footer", default="PAUT Explainable-AI Digital Twin")
    ap.add_argument("--ablation", action="store_true", help="append the ablation curve as Appendix A")
    a = ap.parse_args(argv)
    build(a.md, a.out, a.title, a.subtitle, a.footer, a.ablation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

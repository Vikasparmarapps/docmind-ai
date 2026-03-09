# ============================================================
# export/pdf_export.py  —  Generates styled PDF from Q&A pairs
# ============================================================
# This file has ONE job: take a list of Q&A pairs → return PDF bytes
#
# We use ReportLab — a Python library for creating PDF files.
# It works by building a "story" (list of elements like paragraphs,
# dividers, spacers) and then rendering them onto A4 pages.

import io
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER


def export_pdf(pairs: list, title: str = "DocMind AI — Q&A Export") -> bytes:
    """
    Generate a styled PDF file from Q&A pairs.

    Args:
        pairs : list of {"q": "question", "a": "answer"} dicts
        title : title shown at the top of the PDF

    Returns:
        PDF file as bytes (ready for st.download_button)

    Visual structure of each Q&A entry:
        ┌────────────────────────────────┐
        │  Q 01  (cyan label)            │
        │  The question text here        │
        │  A 01  (green label)           │
        │  The answer text here          │
        │  ─────────────────────────     │
        └────────────────────────────────┘
    """
    buf = io.BytesIO()

    # Create the PDF document (A4 size with margins)
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ── Define text styles (like CSS for PDF)
    style_title = ParagraphStyle(
        "title",
        fontSize=20, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#00e5ff"),
        spaceAfter=4, alignment=TA_CENTER,
    )
    style_meta = ParagraphStyle(
        "meta",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14, alignment=TA_CENTER,
    )
    style_q_label = ParagraphStyle(
        "qlabel",
        fontSize=8, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#00e5ff"),
        spaceBefore=8, spaceAfter=2,
    )
    style_q_text = ParagraphStyle(
        "qtext",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#f1f5f9"),
        spaceAfter=5, leftIndent=10,
    )
    style_a_label = ParagraphStyle(
        "alabel",
        fontSize=8, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#34d399"),
        spaceBefore=2, spaceAfter=2,
    )
    style_a_text = ParagraphStyle(
        "atext",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#94a3b8"),
        spaceAfter=4, leading=14, leftIndent=10,
    )
    style_footer = ParagraphStyle(
        "footer",
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER,
    )

    # ── Build the story (list of elements to render)
    now = datetime.datetime.now().strftime("%B %d, %Y  %I:%M %p")
    story = [
        Paragraph(title, style_title),
        Paragraph(f"Generated: {now}  ·  {len(pairs)} pairs", style_meta),
        HRFlowable(
            width="100%", thickness=1.5,
            color=colors.HexColor("#00e5ff"),
            spaceAfter=14,
        ),
    ]

    # Add each Q&A pair
    for i, pair in enumerate(pairs, 1):
        story += [
            Paragraph(f"Q {i:02d}", style_q_label),
            Paragraph(pair["q"], style_q_text),
            Paragraph(f"A {i:02d}", style_a_label),
            Paragraph(pair["a"].replace("\n", "<br/>"), style_a_text),
            HRFlowable(
                width="100%", thickness=0.4,
                color=colors.HexColor("#1e293b"),
                spaceBefore=6, spaceAfter=6,
            ),
        ]

    # Footer
    story += [
        Spacer(1, 0.4 * cm),
        Paragraph("DocMind AI · RAG · ChromaDB · Ollama · LangChain", style_footer),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()

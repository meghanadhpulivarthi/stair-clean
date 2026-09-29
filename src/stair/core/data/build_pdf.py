import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def write_bookmarked_pdf(path, chapters):
    pdf_canvas = canvas.Canvas(str(path), pagesize=letter)

    for chapter_index, (title, paragraphs) in enumerate(chapters):
        bookmark_key = f"chapter_{chapter_index}"
        pdf_canvas.bookmarkPage(bookmark_key)
        pdf_canvas.addOutlineEntry(title, bookmark_key, level=0)

        y_position = 750
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(72, y_position, title)
        y_position -= 30

        pdf_canvas.setFont("Helvetica", 11)
        for paragraph in paragraphs:
            for line in textwrap.wrap(paragraph, width=95):
                if y_position < 72:
                    pdf_canvas.showPage()
                    pdf_canvas.setFont("Helvetica", 11)
                    y_position = 750
                pdf_canvas.drawString(72, y_position, line)
                y_position -= 16
            y_position -= 8

        pdf_canvas.showPage()

    pdf_canvas.save()

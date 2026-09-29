from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_tiny_bookmarked_pdf(path):
    pdf_canvas = canvas.Canvas(str(path), pagesize=letter)

    chapters = [
        ("Introduction", ["This is the introduction chapter.", "It has two short lines."]),
        ("Getting Started", ["This is the getting started chapter.", "It also has two lines."]),
    ]

    for chapter_index, (title, lines) in enumerate(chapters):
        bookmark_key = f"chapter_{chapter_index}"
        pdf_canvas.bookmarkPage(bookmark_key)
        pdf_canvas.addOutlineEntry(title, bookmark_key, level=0)

        y_position = 750
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(72, y_position, title)
        y_position -= 30

        pdf_canvas.setFont("Helvetica", 11)
        for line in lines:
            pdf_canvas.drawString(72, y_position, line)
            y_position -= 16

        pdf_canvas.showPage()

    pdf_canvas.save()

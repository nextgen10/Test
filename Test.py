import requests
import markdown
from bs4 import BeautifulSoup
from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    Table,
    TableStyle,
    Spacer
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ================= CONFIGURATION =================

GITLAB_URL = "https://gitlab.yourcompany.com/api/v4"
PROJECT_ID = "123"              # CHANGE
ISSUE_IID = "456"               # CHANGE
PRIVATE_TOKEN = "your_token"    # CHANGE

OUTPUT_FILE = "gitlab_issue.pdf"

# ===================================================


def fetch_issue():
    headers = {"PRIVATE-TOKEN": PRIVATE_TOKEN}
    url = f"{GITLAB_URL}/projects/{PROJECT_ID}/issues/{ISSUE_IID}"

    response = requests.get(url, headers=headers, timeout=20)

    if response.status_code != 200:
        raise Exception(f"GitLab API Error: {response.status_code} - {response.text}")

    return response.json()


def markdown_to_html(markdown_text):
    return markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"]
    )


def fetch_image(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception:
        pass
    return None


def build_wrapped_table(table_tag, styles, doc):
    table_data = []

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['BodyText'],
        wordWrap='CJK'  # CRITICAL: prevents overflow
    )

    rows = table_tag.find_all("tr")

    for row in rows:
        row_cells = []
        cols = row.find_all(["td", "th"])

        for col in cols:
            text = col.get_text(strip=True)
            row_cells.append(Paragraph(text, cell_style))

        table_data.append(row_cells)

    if not table_data:
        return None

    available_width = doc.width
    num_cols = len(table_data[0])

    col_width = available_width / num_cols
    col_widths = [col_width] * num_cols

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return table


def generate_pdf(title, soup):
    doc = SimpleDocTemplate(OUTPUT_FILE, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        wordWrap='CJK'  # Prevents long URL overflow
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['BodyText'],
        fontName='Courier',
        fontSize=9,
        wordWrap='CJK'
    )

    elements.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    for tag in soup.find_all(["p", "img", "table", "ul", "ol", "pre"]):

        # ✅ Paragraphs
        if tag.name == "p":
            elements.append(Paragraph(tag.decode_contents(), body_style))
            elements.append(Spacer(1, 6))

        # ✅ Code Blocks
        elif tag.name == "pre":
            elements.append(Paragraph(tag.get_text(), code_style))
            elements.append(Spacer(1, 6))

        # ✅ Images (AUTO-FIT SAFE)
        elif tag.name == "img":
            img_url = tag.get("src")
            img_data = fetch_image(img_url)

            if img_data:
                img = Image(img_data)

                max_width = doc.width
                max_height = 300

                img._restrictSize(max_width, max_height)

                elements.append(img)
                elements.append(Spacer(1, 10))

        # ✅ Tables (NON-OVERFLOW SAFE)
        elif tag.name == "table":
            table = build_wrapped_table(tag, styles, doc)

            if table:
                elements.append(table)
                elements.append(Spacer(1, 12))

        # ✅ Bullet Lists
        elif tag.name == "ul":
            for li in tag.find_all("li"):
                elements.append(Paragraph(f"• {li.get_text()}", body_style))
            elements.append(Spacer(1, 6))

        # ✅ Numbered Lists
        elif tag.name == "ol":
            idx = 1
            for li in tag.find_all("li"):
                elements.append(Paragraph(f"{idx}. {li.get_text()}", body_style))
                idx += 1
            elements.append(Spacer(1, 6))

    doc.build(elements)


def main():
    print("Fetching GitLab issue...")
    issue = fetch_issue()

    title = issue.get("title", "GitLab Issue")
    description = issue.get("description", "")

    print("Converting Markdown → HTML...")
    html_content = markdown_to_html(description)

    soup = BeautifulSoup(html_content, "html.parser")

    print("Generating PDF (overflow-safe)...")
    generate_pdf(title, soup)

    print(f"PDF generated successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

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
from reportlab.lib.styles import getSampleStyleSheet

# ================= CONFIGURATION =================

GITLAB_URL = "https://gitlab.yourcompany.com/api/v4"
PROJECT_ID = "123"          # Change
ISSUE_IID = "456"           # Change
PRIVATE_TOKEN = "your_token"  # Change

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


def parse_table(table_tag):
    table_data = []

    rows = table_tag.find_all("tr")
    for row in rows:
        cols = row.find_all(["td", "th"])
        row_data = [col.get_text(strip=True) for col in cols]
        table_data.append(row_data)

    return table_data


def generate_pdf(title, soup):
    doc = SimpleDocTemplate(OUTPUT_FILE, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    for tag in soup.find_all(["p", "img", "table", "ul", "ol", "pre"]):

        # ✅ Paragraphs
        if tag.name == "p":
            elements.append(Paragraph(tag.decode_contents(), styles['BodyText']))
            elements.append(Spacer(1, 6))

        # ✅ Code blocks
        elif tag.name == "pre":
            elements.append(Paragraph(f"<font name='Courier'>{tag.get_text()}</font>", styles['Code']))
            elements.append(Spacer(1, 6))

        # ✅ Images
        elif tag.name == "img":
            img_url = tag.get("src")
            img_data = fetch_image(img_url)

            if img_data:
                img = Image(img_data)
                img._restrictSize(450, 300)
                elements.append(img)
                elements.append(Spacer(1, 10))

        # ✅ Tables
        elif tag.name == "table":
            table_data = parse_table(tag)

            if table_data:
                table = Table(table_data, repeatRows=1)

                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))

                elements.append(table)
                elements.append(Spacer(1, 12))

        # ✅ Bullet lists
        elif tag.name == "ul":
            for li in tag.find_all("li"):
                elements.append(Paragraph(f"• {li.get_text()}", styles['BodyText']))
            elements.append(Spacer(1, 6))

        # ✅ Numbered lists
        elif tag.name == "ol":
            idx = 1
            for li in tag.find_all("li"):
                elements.append(Paragraph(f"{idx}. {li.get_text()}", styles['BodyText']))
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

    print("Generating PDF...")
    generate_pdf(title, soup)

    print(f"PDF generated successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

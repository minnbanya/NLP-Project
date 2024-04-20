import sys
import json
from os import path, listdir, mkdir
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import string

def clean_product_name(name):
    # Replace "/" with " - "
    name = name.replace("/", " - ")
    # Remove other invalid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    name = ''.join(c for c in name if c in valid_chars)
    return name


def draw_multiline_text(canvas_obj, x, y, text, max_width=400, line_height=14):
    lines = []
    current_line = ""
    words = text.split()
    for word in words:
        if canvas_obj.stringWidth(current_line + " " + word) < max_width:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())

    text_height = len(lines) * line_height  # Calculate total text height

    for line in lines:
        canvas_obj.drawString(x, y, line)
        y -= line_height

    return text_height  # Return the total height of the drawn text


def create_pdf(product_info, output_folder):
    title = clean_product_name(product_info.get('title', 'Unknown'))
    description = product_info.get('description', 'No description available')
    price = product_info.get('price', 'Unknown')
    features = "\n".join(product_info.get('features', []))
    images = product_info.get('images', [])

    folder_name = output_folder
    filename = path.join(folder_name, f"{title}.pdf")
    c = canvas.Canvas(filename, pagesize=letter)

    # Draw title
    c.drawString(100, 750, f"Title: {title}")

    # Draw price
    c.drawString(100, 730, f"Price: {price}")

    # Draw description
    c.drawString(100, 710, "Description:")
    desc_text = "\n".join(description)  
    desc_height = draw_multiline_text(c, 120, 690, desc_text)

    # Draw features
    c.drawString(100, 710 - desc_height - 30, "Features:")
    features_height = draw_multiline_text(c, 120, 710 - desc_height - 50, features)

    # Add images to the PDF
    y_offset = 710 - desc_height - features_height - 300  # Adjust y offset based on text height
    for image_info in images:
        image_url = image_info.get('large', '')
        if image_url:
            img = ImageReader(image_url)
            c.drawImage(img, 100, y_offset, width=200, height=200)
            y_offset -= 220  # Adjust vertical position for the next image

    c.save()


def process_jsonl_files(json_folder):
    for file_name in listdir(json_folder):
        if file_name.endswith(".jsonl"):
            file_path = path.join(json_folder, file_name)
            folder_name = file_name.split("_", 1)[1].split(".")[0]  # Extract folder name from file name
            output_folder = path.join(json_folder, folder_name)
            if not path.exists(output_folder):
                mkdir(output_folder)
            with open(file_path, 'r') as file:
                for line in file:
                    product_info = json.loads(line)
                    create_pdf(product_info, output_folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_json_folder")
        sys.exit(1)

    json_folder = sys.argv[1]

    if not path.exists(json_folder):
        print("Input JSON folder does not exist.")
        sys.exit(1)

    process_jsonl_files(json_folder)

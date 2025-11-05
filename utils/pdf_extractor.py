import fitz  # PyMuPDF

async def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    image_bytes_list = []

    for page in doc:
        all_text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes_list.append(base_image["image"])
    return all_text, image_bytes_list

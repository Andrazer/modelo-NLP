import fitz

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_path, nombre_archivo):
    text = ""
    with fitz.open(f'pdf/{nombre_archivo}') as pdf_document:
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()

    # Guarda el texto en un archivo de texto en la carpeta 'uploads' con el mismo nombre base
    txt_path = f'pdf/{nombre_archivo}.txt'
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)


archivo_name = input("Introduce ruta archivo: ")

extract_text_from_pdf("pdf/", archivo_name)
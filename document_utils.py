import fitz 
import io

def extract_text(file):
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in pdf])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""


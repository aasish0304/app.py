from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import spacy
from spacy.cli import download as spacy_download
from transformers import pipeline
import os

app = FastAPI()

# Load SpaCy model with auto-download
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extract_text(file):
    """Extracts raw text from PDF"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        text = extract_text(file.file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        doc = nlp(text)

        entities = {"dates": [], "money": [], "orgs": [], "persons": []}
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["orgs"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["persons"].append(ent.text)

        summary = summarizer(text[:1000], max_length=150, min_length=60, do_sample=False)[0]['summary_text']

        return {
            "summary": summary,
            "entities": entities,
            "risk": "High" if entities["dates"] else "Unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render port if available
    uvicorn.run(app, host="0.0.0.0", port=port)

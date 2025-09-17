from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader
import spacy
from transformers import pipeline

app = FastAPI()

# Load NLP + summarizer
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text(file):
    """Extracts raw text from PDF"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    text = extract_text(file.file)
    doc = nlp(text)

    entities = {"dates": [], "money": [], "orgs": [], "persons": []}
    for ent in doc.ents:
        if ent.label_ == "DATE": entities["dates"].append(ent.text)
        elif ent.label_ == "MONEY": entities["money"].append(ent.text)
        elif ent.label_ == "ORG": entities["orgs"].append(ent.text)
        elif ent.label_ == "PERSON": entities["persons"].append(ent.text)

    summary = summarizer(text[:1000], max_length=150, min_length=60, do_sample=False)[0]['summary_text']

    return {
        "summary": summary,
        "entities": entities,
        "risk": "High" if entities["dates"] else "Unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

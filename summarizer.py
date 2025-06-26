from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text, max_words=150):
    trimmed = text[:3000]
    result = summarizer(trimmed, max_length=150, min_length=50, do_sample=False)
    return result[0]['summary_text']


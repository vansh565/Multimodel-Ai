import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(chunk):
    doc = nlp(chunk)
    return list(set([ent.text for ent in doc.ents]))
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import json

# Load dataset
with open("data/medical_dataset.json") as f:
    data = json.load(f)

# Create PDF
doc = SimpleDocTemplate("data/medical_dataset.pdf")
styles = getSampleStyleSheet()

content = []

for item in data:
    text = f"""
    <b>Disease:</b> {item['disease']}<br/>
    <b>Symptoms:</b> {', '.join(item['symptoms'])}<br/>
    <b>Description:</b> {item['description']}<br/>
    <b>Treatment:</b> {item['treatment']}<br/>
    <b>Doctor:</b> {item['doctor']['name']} - {item['doctor']['profession']}<br/>
    <br/>
    """

    content.append(Paragraph(text, styles["Normal"]))
    content.append(Spacer(1, 12))

# Build PDF
doc.build(content)

print("✅ PDF Created: data/medical_dataset.pdf")
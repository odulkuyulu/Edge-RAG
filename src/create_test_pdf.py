from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    # Create a PDF document
    c = canvas.Canvas("data/test_ai_initiatives.pdf", pagesize=letter)
    
    # Set font and size
    c.setFont("Helvetica", 12)
    
    # Add content
    c.drawString(100, 750, "AI Initiatives in Healthcare and Finance")
    c.setFont("Helvetica", 10)
    
    # Add paragraphs
    y = 700
    paragraphs = [
        "Amazon Web Services (AWS) has announced significant investments in AI for healthcare and finance sectors.",
        "During his visit to the Dubai office, Jeff Bezos highlighted the launch of new machine learning services.",
        "",
        "Healthcare Initiatives:",
        "- AI-powered diagnostic tools",
        "- Patient data analysis systems",
        "- Automated medical record processing",
        "",
        "Finance Sector Developments:",
        "- Fraud detection systems",
        "- Risk assessment models",
        "- Automated trading algorithms",
        "",
        "These developments are part of AWS's $2 billion investment in AI research and development."
    ]
    
    for para in paragraphs:
        c.drawString(100, y, para)
        y -= 20
    
    # Save the PDF
    c.save()

if __name__ == "__main__":
    create_test_pdf() 
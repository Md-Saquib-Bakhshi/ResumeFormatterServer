import io
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import black, blue, darkgreen
from reportlab.lib.units import inch

def generate_pdf_from_json(parsed_data: Dict[str, Any]) -> bytes:
    """Generates a PDF file from parsed resume JSON data using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define custom paragraph styles for better control
    styles.add(ParagraphStyle(name='ResumeHeading1', fontSize=24, leading=28, alignment=TA_CENTER, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ResumeHeading2', fontSize=16, leading=18, spaceAfter=6, fontName='Helvetica-Bold', textColor=blue))
    styles.add(ParagraphStyle(name='ResumeSubHeading', fontSize=12, leading=14, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ResumeBodyText', fontSize=10, leading=12, spaceAfter=2, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ResumeBullet', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ResumeItalicBody', fontSize=10, leading=12, fontName='Helvetica-Oblique'))
    styles.add(ParagraphStyle(name='ResumeLink', fontSize=10, leading=12, fontName='Helvetica', textColor=darkgreen))

    story = []

    # Basic Details
    basic_details = parsed_data.get("basic_details", {})
    name = basic_details.get("name", "N/A")
    email = basic_details.get("email", "N/A")
    phone = basic_details.get("phone", "N/A")
    links = basic_details.get("links", {})

    story.append(Paragraph(name, styles['ResumeHeading1']))
    contact_info = []
    if email and email != "N/A":
        contact_info.append(email)
    if phone and phone != "N/A":
        contact_info.append(phone)
    if links:
        link_parts = []
        for k, v in links.items():
            if v and v != "N/A":
                link_parts.append(f"{k.capitalize()}: <font color='darkgreen'>{v}</font>")
        if link_parts:
            contact_info.append(" | ".join(link_parts))
    
    if contact_info:
        story.append(Paragraph(" | ".join(contact_info), styles['ResumeBodyText']))
    story.append(Spacer(1, 0.2 * inch))

    # Professional Summary
    summary = parsed_data.get("professional_summary", "N/A")
    if summary and summary != "N/A":
        story.append(Paragraph("Professional Summary", styles['ResumeHeading2']))
        story.append(Paragraph(summary, styles['ResumeBodyText']))
        story.append(Spacer(1, 0.1 * inch))

    # Technical Expertise
    tech_skills = parsed_data.get("technical_expertise", [])
    if tech_skills:
        story.append(Paragraph("Technical Expertise", styles['ResumeHeading2']))
        story.append(Paragraph(", ".join(tech_skills), styles['ResumeBodyText']))
        story.append(Spacer(1, 0.1 * inch))

    # Professional Experience
    experience = parsed_data.get("professional_experience", [])
    if experience:
        story.append(Paragraph("Professional Experience", styles['ResumeHeading2']))
        for exp in experience:
            company = exp.get("company", "N/A")
            role = exp.get("role", "N/A")
            date_range = exp.get("date_range", "N/A")
            client_engagement = exp.get("client_engagement", "N/A")
            program = exp.get("program", "N/A")
            responsibilities = exp.get("responsibilities", [])

            story.append(Paragraph(f"<font name='Helvetica-Bold'>{role}</font> at <font name='Helvetica-Bold'>{company}</font> ({date_range})", styles['ResumeSubHeading']))
            
            if client_engagement and client_engagement != "N/A":
                story.append(Paragraph(f"Client/Project: {client_engagement}", styles['ResumeItalicBody']))
            if program and program != "N/A":
                story.append(Paragraph(f"Program: {program}", styles['ResumeItalicBody']))
            
            if responsibilities:
                for resp in responsibilities:
                    if resp and resp != "N/A":
                        story.append(Paragraph(f"• {resp}", styles['ResumeBullet']))
            story.append(Spacer(1, 0.1 * inch)) # Spacing between experiences

    # Certifications
    certifications = parsed_data.get("certifications", [])
    if certifications:
        story.append(Paragraph("Certifications", styles['ResumeHeading2']))
        for cert in certifications:
            title = cert.get("title", "N/A")
            date = cert.get("date", "N/A")
            if title != "N/A":
                story.append(Paragraph(f"• {title} ({date})", styles['ResumeBullet']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
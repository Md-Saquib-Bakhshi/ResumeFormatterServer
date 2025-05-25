import io
from typing import Dict, Any

from docx import Document
from docx.shared import Inches # Inches is not used in your provided code, but good to keep if intended for future use.

def generate_docx_from_json(parsed_data: Dict[str, Any]) -> bytes:
    """Generates a DOCX file from parsed resume JSON data."""
    document = Document()
    styles = document.styles

    # Basic Details
    basic_details = parsed_data.get("basic_details", {})
    name = basic_details.get("name", "N/A")
    email = basic_details.get("email", "N/A")
    phone = basic_details.get("phone", "N/A")
    links = basic_details.get("links", {})

    document.add_heading(name, level=1)
    if email and email != "N/A":
        document.add_paragraph(f"Email: {email}")
    if phone and phone != "N/A":
        document.add_paragraph(f"Phone: {phone}")
    if links:
        link_str = ", ".join([f"{k.capitalize()}: {v}" for k, v in links.items() if v and v != "N/A"])
        if link_str:
            document.add_paragraph(f"Links: {link_str}")
    document.add_paragraph("") # Add some spacing

    # Professional Summary
    summary = parsed_data.get("professional_summary", "N/A")
    if summary and summary != "N/A":
        document.add_heading("Professional Summary", level=2)
        document.add_paragraph(summary)
        document.add_paragraph("")

    # Technical Expertise
    tech_skills = parsed_data.get("technical_expertise", [])
    if tech_skills:
        document.add_heading("Technical Expertise", level=2)
        document.add_paragraph(", ".join(tech_skills))
        document.add_paragraph("")

    # Professional Experience
    experience = parsed_data.get("professional_experience", [])
    if experience:
        document.add_heading("Professional Experience", level=2)
        for exp in experience:
            company = exp.get("company", "N/A")
            role = exp.get("role", "N/A")
            date_range = exp.get("date_range", "N/A")
            client_engagement = exp.get("client_engagement", "N/A")
            program = exp.get("program", "N/A")
            responsibilities = exp.get("responsibilities", [])

            p = document.add_paragraph()
            p.add_run(f"{role} at {company}").bold = True
            p.add_run(f" ({date_range})")
            
            if client_engagement and client_engagement != "N/A":
                document.add_paragraph(f"Client/Project: {client_engagement}", style='Intense Quote') # Using another built-in style
            if program and program != "N/A":
                document.add_paragraph(f"Program: {program}", style='Intense Quote')
            
            if responsibilities:
                document.add_paragraph("Responsibilities:")
                for resp in responsibilities:
                    if resp and resp != "N/A":
                        document.add_paragraph(resp, style='List Bullet')
            document.add_paragraph("") # Spacing between experiences

    # Certifications
    certifications = parsed_data.get("certifications", [])
    if certifications:
        document.add_heading("Certifications", level=2)
        for cert in certifications:
            title = cert.get("title", "N/A")
            date = cert.get("date", "N/A")
            if title != "N/A":
                document.add_paragraph(f"{title} ({date})", style='List Bullet')
        document.add_paragraph("")

    # Save document to bytes
    bio = io.BytesIO()
    document.save(bio)
    bio.seek(0)
    return bio.getvalue()
import io
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, Paragraph, Spacer, Frame, PageBreak, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import black, blue, darkgreen, HexColor, white
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepTogether

# --- Custom Document Template for Persistent Header and Multi-Column Layout ---
class ResumeDocTemplate(BaseDocTemplate):
    def __init__(self, filename, parsed_data, **kw):
        self.parsed_data = parsed_data
        BaseDocTemplate.__init__(self, filename, **kw)
        self.pagesize = letter

        page_width, page_height = letter
        left_margin = kw.get('leftMargin', 0.5 * inch)
        right_margin = kw.get('rightMargin', 0.5 * inch)
        top_margin = kw.get('topMargin', 0.5 * inch)
        bottom_margin = kw.get('bottomMargin', 0.5 * inch)

        usable_width = page_width - left_margin - right_margin
        usable_height = page_height - top_margin - bottom_margin

        header_actual_height = 100
        spacer_below_header = 0.2 * inch
        total_header_block_height = header_actual_height + spacer_below_header

        frame_y_top = page_height - top_margin - total_header_block_height
        frame_y_bottom = bottom_margin
        frame_height = frame_y_top - frame_y_bottom

        left_frame_width = usable_width * 0.35
        right_frame_width = usable_width * 0.65

        # Two column frames for first pages
        frame1 = Frame(left_margin, frame_y_bottom, left_frame_width, frame_height,
                       leftPadding=5, bottomPadding=5, rightPadding=5, topPadding=5,
                       showBoundary=0, id='left_column_frame')

        frame2 = Frame(left_margin + left_frame_width, frame_y_bottom, right_frame_width, frame_height,
                       leftPadding=5, bottomPadding=5, rightPadding=5, topPadding=5,
                       showBoundary=0, id='right_column_frame')
        
        # Single column frame for experience pages
        full_width_frame = Frame(left_margin, frame_y_bottom, usable_width, frame_height,
                                 leftPadding=5, bottomPadding=5, rightPadding=5, topPadding=5,
                                 showBoundary=0, id='full_width_frame')

        self.addPageTemplates([
            PageTemplate(id='TwoColPage', frames=[frame1, frame2], onPage=self._header_on_page),
            PageTemplate(id='SingleColPage', frames=[full_width_frame], onPage=self._header_on_page)
        ])

    def _header_on_page(self, canvas, doc):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='HeaderLogo', fontSize=32, leading=36, alignment=TA_LEFT, fontName='Helvetica-Bold', textColor=white))
        styles.add(ParagraphStyle(name='HeaderName', fontSize=36, leading=40, alignment=TA_RIGHT, fontName='Helvetica-Bold', textColor=white))

        basic_details = doc.parsed_data.get("basic_details", {})
        name = basic_details.get("name", "Your Name")

        header_height = 100
        header_x = doc.leftMargin
        header_y = doc.pagesize[1] - doc.topMargin - header_height
        header_width = doc.width

        canvas.setFillColor(black)
        canvas.rect(header_x, header_y, header_width, header_height, fill=1)

        logo_para = Paragraph("LOGO", styles['HeaderLogo'])
        name_para = Paragraph(name, styles['HeaderName'])

        logo_para_width, logo_para_height = logo_para.wrapOn(canvas, header_width * 0.35 - 20, header_height)
        name_para_width, name_para_height = name_para.wrapOn(canvas, header_width * 0.65 - 20, header_height)

        logo_x = header_x + 10
        name_x = header_x + header_width - name_para_width - 10

        logo_y = header_y + (header_height - logo_para_height) / 2
        name_y = header_y + (header_height - name_para_height) / 2

        logo_para.drawOn(canvas, logo_x, logo_y)
        name_para.drawOn(canvas, name_x, name_y)


def generate_pdf_from_json(parsed_data: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    
    doc = ResumeDocTemplate(buffer, parsed_data,
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='ResumeHeading1', fontSize=24, leading=28, alignment=TA_CENTER, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ResumeHeading2', fontSize=16, leading=18, spaceAfter=6, fontName='Helvetica-Bold', textColor=blue))
    styles.add(ParagraphStyle(name='ResumeSubHeading', fontSize=12, leading=14, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ResumeBodyText', fontSize=10, leading=12, spaceAfter=2, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ResumeBullet', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ResumeItalicBody', fontSize=10, leading=12, fontName='Helvetica-Oblique'))
    styles.add(ParagraphStyle(name='ResumeLink', fontSize=10, leading=12, fontName='Helvetica', textColor=darkgreen))
    styles.add(ParagraphStyle(name='TechCategoryHeading', fontSize=11, leading=13, spaceAfter=2, fontName='Helvetica-Bold', textColor=HexColor('#333333')))
    styles.add(ParagraphStyle(name='ExperienceCompany', fontSize=14, leading=16, spaceAfter=2, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ExperienceDetails', fontSize=11, leading=13, spaceAfter=2, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ResponsibilitiesHeader', fontSize=11, leading=13, spaceAfter=4, fontName='Helvetica-Bold'))

    story = []

    # Start with two-column layout
    story.append(PageBreak())
    story[-1].pageTemplate = 'TwoColPage'

    # LEFT COLUMN CONTENT
    
    # Education
    education = parsed_data.get("education", [])
    if education:
        story.append(Paragraph("Education", styles['ResumeHeading2']))
        for edu_item in education:
            if isinstance(edu_item, dict):
                degree = edu_item.get("degree", "N/A")
                institution = edu_item.get("institution", "N/A")
                date_range = edu_item.get("date_range", "N/A")
                edu_text = f"• {degree}"
                if institution and institution != "N/A":
                    edu_text += f" at {institution}"
                if date_range and date_range != "N/A":
                    edu_text += f" ({date_range})"
            else:
                edu_text = f"• {str(edu_item).strip()}"

            if edu_text.strip() != "•":
                story.append(Paragraph(edu_text, styles['ResumeBullet']))
        story.append(Spacer(1, 0.15 * inch))

    # Technical Expertise
    tech_expertise_data = parsed_data.get("technical_expertise", {})
    if tech_expertise_data:
        story.append(Paragraph("Technical Expertise", styles['ResumeHeading2']))
        sorted_categories = sorted(tech_expertise_data.keys())
        for category in sorted_categories:
            skills = tech_expertise_data.get(category, [])
            if skills:
                story.append(Paragraph(f"<b>{category}:</b>", styles['TechCategoryHeading']))
                for skill in skills:
                    if skill and skill.strip() != "N/A":
                        story.append(Paragraph(f"• {skill.strip()}", styles['ResumeBullet']))
                story.append(Spacer(1, 0.05 * inch))
        story.append(Spacer(1, 0.15 * inch))

    # Certifications
    certifications = parsed_data.get("certifications", [])
    if certifications:
        story.append(Paragraph("Certifications", styles['ResumeHeading2']))
        for cert in certifications:
            if isinstance(cert, dict):
                title = cert.get("title", "N/A")
                date = cert.get("date", "N/A")
                if title != "N/A":
                    cert_text = f"• {title}"
                    if date and date != "N/A":
                        cert_text += f" ({date})"
                    story.append(Paragraph(cert_text, styles['ResumeBullet']))
            else:
                story.append(Paragraph(f"• {str(cert).strip()}", styles['ResumeBullet']))
        story.append(Spacer(1, 0.15 * inch))

    # RIGHT COLUMN CONTENT
    
    # Profile Summary (Professional Summary)
    summary_lines = parsed_data.get("professional_summary", [])
    if summary_lines:
        story.append(Paragraph("Profile Summary", styles['ResumeHeading2']))
        for line in summary_lines:
            if line and line.strip() != "N/A":
                story.append(Paragraph(f"• {line.strip()}", styles['ResumeBullet']))
        story.append(Spacer(1, 0.15 * inch))

    # EXPERIENCE SECTION (New page, single column)
    experience = parsed_data.get("professional_experience", [])
    if experience:
        story.append(PageBreak())
        story[-1].pageTemplate = 'SingleColPage'
        
        story.append(Paragraph("Professional Experience", styles['ResumeHeading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        for exp in experience:
            company = exp.get("company", "N/A")
            role = exp.get("role", "N/A")
            date_range = exp.get("date_range", "N/A")
            client_engagement = exp.get("client_engagement", "")
            program = exp.get("program", "")
            responsibilities = exp.get("responsibilities", [])

            exp_block = []
            
            # Company name
            exp_block.append(Paragraph(f"Company: {company}", styles['ExperienceCompany']))
            
            # Date range
            if date_range and date_range != "N/A":
                exp_block.append(Paragraph(f"Date: {date_range}", styles['ExperienceDetails']))
            
            # Role
            if role and role != "N/A":
                exp_block.append(Paragraph(f"Role: {role}", styles['ExperienceDetails']))
            
            # Client Engagement
            if client_engagement and client_engagement.strip() != "N/A" and client_engagement.strip():
                exp_block.append(Paragraph(f"Client Engagement: {client_engagement}", styles['ExperienceDetails']))
            
            # Program
            if program and program.strip() != "N/A" and program.strip():
                exp_block.append(Paragraph(f"Program: {program}", styles['ExperienceDetails']))
            
            # Responsibilities
            if responsibilities:
                exp_block.append(Paragraph("RESPONSIBILITIES:", styles['ResponsibilitiesHeader']))
                for resp in responsibilities:
                    if resp and resp.strip() != "N/A":
                        exp_block.append(Paragraph(f"• {resp.strip()}", styles['ResumeBullet']))
            
            exp_block.append(Spacer(1, 0.2 * inch))
            story.append(KeepTogether(exp_block))

    # Build the PDF document
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
import io
from typing import Dict, Any

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, FrameBreak, NextPageTemplate, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

# --- Custom Document Template for the new design ---
class NewResumeDocTemplate(BaseDocTemplate):
    def __init__(self, filename, parsed_data, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        self.parsed_data = parsed_data
        self.pagesize = LETTER
        self.logo_path = kw.get('logo_path', None)

        width, height = self.pagesize
        
        self.HEADER_HEIGHT = 120

        self.CONTAINER_MARGIN_LR = 5

        self.desired_header_to_container_gap = 10
        self.LEFT_PAGE_PADDING = 15
        self.RIGHT_PAGE_PADDING = 15
        self.FRAME_INNER_PADDING = 5

        self.LEFT_CONTAINER_WIDTH_PERCENT = 0.35
        self.RIGHT_CONTAINER_WIDTH_PERCENT = 0.65

        self.container_top_y_for_frames = height - self.HEADER_HEIGHT - self.desired_header_to_container_gap
        self.container_height_for_frames = self.container_top_y_for_frames - self.bottomMargin

        self.total_container_width = width - self.LEFT_PAGE_PADDING - self.RIGHT_PAGE_PADDING
        self.left_container_width_for_frame = self.total_container_width * self.LEFT_CONTAINER_WIDTH_PERCENT
        self.right_container_width_for_frame = self.total_container_width * self.RIGHT_CONTAINER_WIDTH_PERCENT

        # Frames for the first page
        left_frame = Frame(
            self.LEFT_PAGE_PADDING + self.FRAME_INNER_PADDING,
            self.bottomMargin + self.FRAME_INNER_PADDING,
            self.left_container_width_for_frame - (2 * self.FRAME_INNER_PADDING),
            self.container_height_for_frames - (2 * self.FRAME_INNER_PADDING),
            id='left',
            showBoundary=0
        )

        right_frame = Frame(
            self.LEFT_PAGE_PADDING + self.left_container_width_for_frame + self.CONTAINER_MARGIN_LR + self.FRAME_INNER_PADDING,
            self.bottomMargin + self.FRAME_INNER_PADDING,
            self.right_container_width_for_frame - (2 * self.FRAME_INNER_PADDING),
            self.container_height_for_frames - (2 * self.FRAME_INNER_PADDING),
            id='right',
            showBoundary=0
        )

        # Frame for the experience page (full width, standard margins)
        self.EXPERIENCE_CONTENT_PADDING = 0.5 * inch
        
        experience_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            width - self.leftMargin - self.rightMargin,
            height - self.topMargin - self.bottomMargin,
            id='experience_frame',
            leftPadding=self.EXPERIENCE_CONTENT_PADDING,
            rightPadding=self.EXPERIENCE_CONTENT_PADDING,
            topPadding=self.EXPERIENCE_CONTENT_PADDING,
            bottomPadding=self.EXPERIENCE_CONTENT_PADDING,
            showBoundary=0
        )

        # Define the PageTemplates
        self.addPageTemplates([
            PageTemplate(id='FirstPage', frames=[left_frame, right_frame], onPage=self._first_page_elements),
            PageTemplate(id='ExperiencePage', frames=[experience_frame], onPage=self._experience_page_elements)
        ])

    def _first_page_elements(self, canvas: Canvas, doc):
        width, height = doc.pagesize
        
        # === HEADER SECTION ===
        canvas.setFillColor(colors.black)
        canvas.setLineWidth(0)
        canvas.rect(0, height - self.HEADER_HEIGHT, width, self.HEADER_HEIGHT, fill=1)

        logo_section_width = width * 0.3
        try:
            if self.logo_path:
                logo_max_width = logo_section_width * 0.8
                logo_max_height = self.HEADER_HEIGHT * 0.7
                logo_x = (logo_section_width - logo_max_width) / 2
                logo_y = height - self.HEADER_HEIGHT + (self.HEADER_HEIGHT - logo_max_height) / 2
                canvas.drawImage(ImageReader(self.logo_path), logo_x, logo_y,
                                    width=logo_max_width, height=logo_max_height,
                                    preserveAspectRatio=True, mask='auto')
            else:
                canvas.setFont("Helvetica", 12)
                canvas.setFillColor(colors.white)
                placeholder_text = "LOGO"
                text_width = canvas.stringWidth(placeholder_text, "Helvetica", 12)
                placeholder_x = (logo_section_width - text_width) / 2
                placeholder_y = height - self.HEADER_HEIGHT/2 - 6
                canvas.drawString(placeholder_x, placeholder_y, placeholder_text)
        except Exception:
            canvas.setFont("Helvetica", 12)
            canvas.setFillColor(colors.white)
            placeholder_text = "LOGO"
            text_width = canvas.stringWidth(placeholder_text, "Helvetica", 12)
            placeholder_x = (logo_section_width - text_width) / 2
            placeholder_y = height - self.HEADER_HEIGHT/2 - 6
            canvas.drawString(placeholder_x, placeholder_y, placeholder_text)

        # Header Name
        canvas.setFont("Helvetica-Bold", 20)
        canvas.setFillColor(colors.white)
        name_text = self.parsed_data.get("basic_details", {}).get("name", "Your Name").upper()
        name_x = logo_section_width + 20
        name_y = height - self.HEADER_HEIGHT/2 - 8
        canvas.drawString(name_x, name_y, name_text)

        # === CONTAINER SECTION ===
        container_top_y = height - self.HEADER_HEIGHT - self.desired_header_to_container_gap
        container_height_calc = container_top_y - doc.bottomMargin

        left_container_x = self.LEFT_PAGE_PADDING
        total_container_width = width - self.LEFT_PAGE_PADDING - self.RIGHT_PAGE_PADDING
        left_container_width = total_container_width * self.LEFT_CONTAINER_WIDTH_PERCENT
        right_container_x = left_container_x + left_container_width + self.CONTAINER_MARGIN_LR

        canvas.setLineWidth(0)

        # Draw left container background
        canvas.setFillColor(colors.HexColor("#006b64"))
        canvas.rect(left_container_x, doc.bottomMargin,
                                    left_container_width,
                                    container_height_calc, fill=1, stroke=0)

        # Draw right container background
        canvas.setFillColor(colors.white)
        canvas.rect(right_container_x, doc.bottomMargin,
                                    self.right_container_width_for_frame,
                                    container_height_calc, fill=1, stroke=0)

    def _experience_page_elements(self, canvas: Canvas, doc):
        pass

def generate_pdf_from_json(parsed_data: Dict[str, Any], logo_path: str = None) -> bytes:
    """
    Generates a PDF resume from parsed JSON data using a custom design.

    Args:
        parsed_data (Dict[str, Any]): A dictionary containing the parsed resume information.
        logo_path (str, optional): Path to a logo image file. If provided, the logo will be
                                    displayed in the header. Defaults to None.

    Returns:
        bytes: The generated PDF content as bytes.
    """
    buffer = io.BytesIO()
    
    doc = NewResumeDocTemplate(buffer, parsed_data,
                                leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                                topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                                logo_path=logo_path)

    styles = getSampleStyleSheet()

    # Left container styles
    styles.add(ParagraphStyle(name="LeftHeading", fontSize=14.5, leading=16.5, textColor=colors.white,
                              spaceAfter=8, spaceBefore=0, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="LeftBullet", fontSize=11, leading=14, textColor=colors.white,
                              leftIndent=0, spaceAfter=3))
    styles.add(ParagraphStyle(name="LeftText", fontSize=10, leading=12, textColor=colors.white,
                              leftIndent=12, spaceAfter=2))
    
    styles.add(ParagraphStyle(name="EducationCombined", fontSize=11, leading=14, textColor=colors.white,
                              leftIndent=0, spaceAfter=6))

    # Right container styles
    styles.add(ParagraphStyle(name="RightHeading", fontSize=14.5, leading=16.5, textColor=colors.black,
                              spaceAfter=8, spaceBefore=0, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="RightText", fontSize=11, leading=14, textColor=colors.black,
                              spaceAfter=4))

    # New styles for the experience page
    styles.add(ParagraphStyle(name="ExperienceHeading", fontSize=14.5, leading=16.5, textColor=colors.black,
                              spaceAfter=8, spaceBefore=0, fontName="Helvetica-Bold"))
    
    styles.add(ParagraphStyle(name="ExperienceCompanyName", fontSize=13, leading=15, textColor=colors.black,
                              spaceAfter=2, fontName="Helvetica-Bold", leftIndent=0))
    
    styles.add(ParagraphStyle(name="ExperienceDetailValue", fontSize=11, leading=13, textColor=colors.black,
                              spaceAfter=6, leftIndent=0))
    
    styles.add(ParagraphStyle(name="ExperienceResponsibilitiesHeading", fontSize=11, leading=13, textColor=colors.black,
                              spaceAfter=4, spaceBefore=6, fontName="Helvetica-Bold"))
    
    styles.add(ParagraphStyle(name="ExperienceResponsibility", fontSize=11, leading=13, textColor=colors.black,
                              leftIndent=12, spaceAfter=2))

    story = []

    # Ensure the document starts with the first page template
    story.append(NextPageTemplate('FirstPage'))

    # Education
    education = parsed_data.get("education", [])
    if education:
        story.append(Paragraph("EDUCATION", styles["LeftHeading"]))
        for edu in education:
            degree = edu.get("degree", "N/A")
            institution = edu.get("institution", "N/A")
            date_range = edu.get("date_range", "N/A")
            
            education_text_parts = [degree]
            if institution and institution.strip() and institution.strip() != "N/A":
                education_text_parts.append(institution)
            if date_range and date_range.strip() and date_range.strip() != "N/A":
                education_text_parts.append(f"({date_range})")
            
            combined_text = " – ".join([part for part in education_text_parts if part and part.strip() != "N/A"])
            if combined_text.strip():
                story.append(Paragraph(combined_text, styles["EducationCombined"]))
        story.append(Spacer(1, 6))

    # Technical Expertise
    technical_expertise = parsed_data.get("technical_expertise", [])
    if technical_expertise:
        story.append(Paragraph("TECHNICAL EXPERTISE", styles["LeftHeading"]))
        for category_item in technical_expertise:
            category_name = category_item.get("category", "N/A")
            skills = category_item.get("skills", [])
            if category_name and category_name.strip() != "N/A" and skills:
                filtered_skills = [skill.strip() for skill in skills if skill and skill.strip() != "N/A"]
                if filtered_skills:
                    skills_formatted = ", ".join([f'"{skill}"' for skill in filtered_skills])
                    story.append(Paragraph(f"• {category_name}: {skills_formatted}", styles["LeftBullet"]))
        story.append(Spacer(1, 10))

    # Certifications
    certifications = parsed_data.get("certifications", [])
    if certifications:
        story.append(Paragraph("CERTIFICATIONS", styles["LeftHeading"]))
        for cert in certifications:
            title = cert.get("title", "N/A")
            date = cert.get("date", "N/A")
            if title and title.strip() != "N/A":
                cert_text = f"• {title.strip()}"
                if date and date.strip() != "N/A":
                    cert_text += f" ({date.strip()})"
                story.append(Paragraph(cert_text, styles["LeftBullet"]))
        story.append(Spacer(1, 10))

    story.append(FrameBreak()) # Break to the right frame on the first page

    # === RIGHT CONTAINER (65%) - Content for First Page ===

    # Professional Summary - APPLY CHARACTER LIMIT HERE
    professional_summary_lines = parsed_data.get("professional_summary", [])
    if professional_summary_lines:
        story.append(Paragraph("PROFESSIONAL SUMMARY", styles["RightHeading"]))
        
        combined_summary_text = ""
        for line in professional_summary_lines:
            if line and line.strip() != "N/A":
                # Add a space before appending to join lines, or just append for first line
                current_line_text = line.strip()
                if combined_summary_text:
                    proposed_text = combined_summary_text + " " + current_line_text
                else:
                    proposed_text = current_line_text

                # Check if adding this line exceeds the character limit
                if len(proposed_text) <= 2058:
                    combined_summary_text = proposed_text
                else:
                    # If it exceeds, truncate the last line if it can fit partially
                    remaining_space = 2058 - len(combined_summary_text)
                    if remaining_space > 0:
                        truncated_line = current_line_text[:remaining_space].strip()
                        if truncated_line: # Only add if there's content after truncation
                            combined_summary_text += " " + truncated_line
                    break # Stop adding more lines

        # Now, split the combined text back into lines for bullet points
        # This is a simple split; for more advanced handling of bullet points
        # within the limit, you might need a more sophisticated text wrapping logic.
        # For simplicity here, we'll re-bullet the combined text if it was too long.
        
        # If the combined summary text is within the limit, break it back into bullet points
        # to maintain the original bullet structure as much as possible.
        # A simple way to represent truncated bullet points is to just append the final text.
        
        # For professional summary, it's often a cohesive paragraph. Let's aim for that.
        # If the original JSON intended bullets for summary, we'll revert to that.
        # Given your original request was "list of lines" for summary, let's keep bullets.
        
        # Re-iterate through the original lines, but only those that fit
        current_char_count = 0
        for line in professional_summary_lines:
            bullet_line = f"• {line.strip()}"
            if line and line.strip() != "N/A":
                # Check character count including the bullet and space
                # Assuming Paragraph object will add some padding/margins,
                # this is a rough estimate for display.
                # It's better to check the raw text length.
                
                # Check if adding this line would exceed the limit
                # We need to account for the bullet and space.
                if current_char_count + len(bullet_line) <= 2058:
                    story.append(Paragraph(bullet_line, styles["RightText"]))
                    current_char_count += len(bullet_line)
                else:
                    # If it's the first line and already too long, truncate it.
                    # Otherwise, stop adding lines.
                    if current_char_count == 0: # First line, needs truncation
                        truncated_line = bullet_line[:2058]
                        story.append(Paragraph(truncated_line, styles["RightText"]))
                        current_char_count = 2058
                    break # Stop adding more lines after the limit is reached

        story.append(Spacer(1, 10))

    # --- Transition to ExperiencePage for Professional Experience if there's content ---
    if parsed_data.get("professional_experience"):
        story.append(NextPageTemplate('ExperiencePage'))
        story.append(PageBreak())

    # === PROFESSIONAL EXPERIENCE - Content for Second (and subsequent) Page(s) ===
    professional_experience = parsed_data.get("professional_experience", [])
    if professional_experience:
        story.append(Paragraph("PROFESSIONAL EXPERIENCE", styles["ExperienceHeading"]))
        story.append(Spacer(1, 0.1 * inch))
        
        for exp in professional_experience:
            # Company Name
            company = exp.get("company", "N/A")
            if company and company.strip() != "N/A":
                story.append(Paragraph(f"<font name='Helvetica-Bold'>Company:</font> {company.strip()}", styles["ExperienceCompanyName"]))
            
            # Date
            date_range = exp.get("date_range", "N/A")
            if date_range and date_range.strip() != "N/A":
                story.append(Paragraph(f"<font name='Helvetica-Bold'>Date:</font> {date_range.strip()}", styles["ExperienceDetailValue"]))
            
            # Role
            role = exp.get("role", "N/A")
            if role and role.strip() != "N/A":
                story.append(Paragraph(f"<font name='Helvetica-Bold'>Role:</font> {role.strip()}", styles["ExperienceDetailValue"]))
            
            # Client Engagement (only if present and not N/A)
            client_engagement = exp.get('client_engagement', "")
            if client_engagement and client_engagement.strip() not in ["", "N/A"]:
                story.append(Paragraph(f"<font name='Helvetica-Bold'>Client Engagement:</font> {client_engagement.strip()}", styles["ExperienceDetailValue"]))

            # Program (only if present and not N/A)
            program = exp.get('program', "")
            if program and program.strip() not in ["", "N/A"]:
                story.append(Paragraph(f"<font name='Helvetica-Bold'>Program:</font> {program.strip()}", styles["ExperienceDetailValue"]))

            # Responsibilities
            responsibilities = exp.get("responsibilities", [])
            if responsibilities:
                story.append(Paragraph("Responsibilities:", styles["ExperienceResponsibilitiesHeading"]))
                for responsibility in responsibilities:
                    if responsibility and responsibility.strip() != "N/A":
                        story.append(Paragraph(f"• {responsibility.strip()}", styles["ExperienceResponsibility"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
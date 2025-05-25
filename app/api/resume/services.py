import io
import json
import asyncio
import concurrent.futures
import re

from PIL import Image
import numpy as np
import fitz # PyMuPDF
import easyocr
from docx import Document # For .docx files
from pydocx import PyDocX # For .doc files

# Import the configured LLM client and prompt from utils
from .llm_config import azure_openai_client, AZURE_DEPLOYMENT_NAME, LLM_RESUME_PARSING_PROMPT

# Initialize EasyOCR reader once globally for performance.
reader = easyocr.Reader(['en'])

# Initialize a global thread pool for running synchronous OCR and LLM calls.
ocr_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

async def _call_llm_for_resume_parsing(resume_text: str) -> dict:
    """
    Internal helper to call Azure OpenAI for structured resume data extraction.
    This function is designed to be run within a thread pool executor.
    """
    # Check if LLM client was successfully initialized
    if azure_openai_client is None:
        raise RuntimeError("Azure OpenAI client is not initialized. Please set up API credentials.")

    loop = asyncio.get_event_loop()
    
    prompt_messages = [
        {"role": "system", "content": LLM_RESUME_PARSING_PROMPT},
        {"role": "user", "content": f"Resume Text:\n\n{resume_text}\n\nExtract the structured resume data:"}
    ]

    try:
        response = await loop.run_in_executor(
            ocr_llm_executor,
            lambda: azure_openai_client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=prompt_messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2500
            )
        )
        
        llm_output_str = response.choices[0].message.content
        llm_data = json.loads(llm_output_str)
        
        # --- Post-processing to ensure adherence to schema and apply defaults ---
        parsed_resume = {
            "basic_details": {
                "name": "N/A", "email": "N/A", "phone": "N/A",
                "links": {}
            },
            "technical_expertise": [],
            "certifications": [],
            "professional_summary": "N/A",
            "professional_experience": []
        }

        if "basic_details" in llm_data and isinstance(llm_data["basic_details"], dict):
            parsed_resume["basic_details"]["name"] = llm_data["basic_details"].get("name", "N/A")
            parsed_resume["basic_details"]["email"] = llm_data["basic_details"].get("email", "N/A")
            parsed_resume["basic_details"]["phone"] = llm_data["basic_details"].get("phone", "N/A")
            if "links" in llm_data["basic_details"] and isinstance(llm_data["basic_details"]["links"], dict):
                for link_key, link_val in llm_data["basic_details"]["links"].items():
                    if link_val and str(link_val).strip() != "N/A":
                        parsed_resume["basic_details"]["links"][link_key.lower()] = str(link_val).strip()

        if "technical_expertise" in llm_data and isinstance(llm_data["technical_expertise"], list):
            parsed_resume["technical_expertise"] = sorted(list(set([
                str(s).strip() for s in llm_data["technical_expertise"] if str(s).strip() != "N/A"
            ])))

        if "certifications" in llm_data and isinstance(llm_data["certifications"], list):
            certs = []
            for cert_item in llm_data["certifications"]:
                if isinstance(cert_item, dict):
                    title = cert_item.get("title", "N/A")
                    date = cert_item.get("date", "N/A")
                    if title and title != "N/A":
                        certs.append({"title": str(title).strip(), "date": str(date).strip()})
                elif isinstance(cert_item, str) and str(cert_item).strip() != "N/A":
                    certs.append({"title": str(cert_item).strip(), "date": "N/A"})
            seen_titles = set()
            unique_certs = []
            for cert in certs:
                if cert["title"].lower() not in seen_titles:
                    unique_certs.append(cert)
                    seen_titles.add(cert["title"].lower())
            parsed_resume["certifications"] = sorted(unique_certs, key=lambda x: x["title"].lower())

        if "professional_summary" in llm_data:
            parsed_resume["professional_summary"] = str(llm_data["professional_summary"]).strip() or "N/A"
        
        experiences = []
        if "professional_experience" in llm_data and isinstance(llm_data["professional_experience"], list):
            for exp in llm_data["professional_experience"]:
                if isinstance(exp, dict):
                    responsibilities = exp.get("responsibilities", [])
                    if isinstance(responsibilities, str):
                        responsibilities = [responsibilities]
                    
                    experiences.append({
                        "company": exp.get("company", "N/A"),
                        "date_range": exp.get("date_range", "N/A"),
                        "role": exp.get("role", "N/A"),
                        "client_engagement": exp.get("client_engagement", "N/A"),
                        "program": exp.get("program", "N/A"),
                        "responsibilities": [str(r).strip() for r in responsibilities if str(r).strip() != "N/A"]
                    })
        
        parsed_resume["professional_experience"] = sorted(
            experiences, 
            key=lambda x: len(x.get("responsibilities", [])), 
            reverse=True
        )
        
        return parsed_resume

    except json.JSONDecodeError as e:
        print(f"ERROR: LLM did not return valid JSON: {llm_output_str if 'llm_output_str' in locals() else 'N/A'} - {e}")
        return {
            "basic_details": {"name": "N/A", "email": "N/A", "phone": "N/A", "links": {}},
            "technical_expertise": [],
            "certifications": [],
            "professional_summary": "N/A",
            "professional_experience": []
        }
    except Exception as e:
        print(f"ERROR: Failed to call Azure OpenAI or process its response: {e}")
        return {
            "basic_details": {"name": "N/A", "email": "N/A", "phone": "N/A", "links": {}},
            "technical_expertise": [],
            "certifications": [],
            "professional_summary": "N/A",
            "professional_experience": []
        }


def _extract_text_from_docx(doc_bytes: bytes) -> str:
    """
    Extracts plain text from a .docx file using python-docx.
    Note: This does not preserve layout or extract text from images within the docx.
    """
    text_content = []
    try:
        document = Document(io.BytesIO(doc_bytes))
        for para in document.paragraphs:
            text_content.append(para.text)
        # Also iterate through tables
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        text_content.append(paragraph.text)
    except Exception as e:
        print(f"ERROR: Failed to extract text from DOCX: {e}")
        raise ValueError(f"Could not read DOCX file: {e}")
    return "\n".join(text_content)


def _extract_text_from_doc(doc_bytes: bytes) -> str:
    """
    Extracts plain text from a .doc file using pydocx.
    Note: This converts .doc to HTML internally and then extracts text.
    It does not preserve layout or extract text from images within the doc.
    """
    text_content = "" # Initialize as empty string
    try:
        html_content = PyDocX.to_html(io.BytesIO(doc_bytes))
        
        cleanr = re.compile('<.*?>')
        text_content = re.sub(cleanr, '', html_content)
        
    except Exception as e:
        print(f"ERROR: Failed to extract text from DOC: {e}")
        raise ValueError(f"Could not read DOC file: {e}")
    return text_content


async def process_resume_document(file_bytes: bytes, original_filename: str, content_type: str) -> dict:
    """
    Handles the entire pipeline for any document type:
    - If PDF: extracts native text and performs OCR on images within the PDF.
    - If DOCX/DOC: extracts plain text directly (no OCR on embedded images).
    - Combines all extracted text.
    - Calls the LLM to parse the combined text into structured JSON.
    - Generates a filename based on the extracted name.
    """
    all_extracted_text_segments = []

    # Determine processing path based on content type
    if content_type == "application/pdf":
        print(f"Processing {original_filename} as PDF...")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        loop = asyncio.get_event_loop()
        ocr_tasks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            text_from_pdf_layer = page.get_text("text")
            if text_from_pdf_layer.strip():
                all_extracted_text_segments.append(f"--- Page {page_num + 1} (PDF Text Layer) ---\n")
                all_extracted_text_segments.append(text_from_pdf_layer)
            
            # Conditional Full-Page OCR Logic
            perform_full_page_ocr = False
            native_text_len = len(text_from_pdf_layer.strip())

            if native_text_len < 100:
                perform_full_page_ocr = True
            else:
                common_section_keywords = ["experience", "education", "skills", "certifications", "summary", "projects", "awards", "licenses"]
                if not any(keyword in text_from_pdf_layer.lower() for keyword in common_section_keywords):
                    perform_full_page_ocr = True 

            # Embedded Image OCR
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image")
                img_ext = base_image.get("ext")

                if not img_bytes:
                    print(f"WARNING: No image bytes extracted for embedded image {img_index} on page {page_num}.")
                    continue

                if img_ext.lower() in ["png", "jpeg", "jpg", "gif", "bmp"]:
                    try:
                        image = Image.open(io.BytesIO(img_bytes))
                        image_np = np.array(image)
                        
                        ocr_task_future = loop.run_in_executor(
                            ocr_llm_executor,
                            reader.readtext,
                            image_np
                        )
                        ocr_tasks.append(ocr_task_future)
                        all_extracted_text_segments.append(f"--- Page {page_num + 1}, Embedded Image {img_index + 1} (OCR Scheduled) ---\n")
                    except Exception as img_e:
                        print(f"ERROR: Failed to prepare embedded image {img_index} on page {page_num} for OCR: {img_e}")
                else:
                    print(f"WARNING: Skipping unsupported embedded image format '{img_ext}' on page {page_num}, image {img_index}.")

            # Full Page OCR (Conditional)
            if perform_full_page_ocr:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    img_bytes_from_page = pix.tobytes("png")

                    if not img_bytes_from_page:
                        print(f"WARNING: No bytes generated when rendering page {page_num} for full page OCR.")
                        continue

                    image_from_page = Image.open(io.BytesIO(img_bytes_from_page))
                    image_from_page_np = np.array(image_from_page)

                    ocr_task_future = loop.run_in_executor(
                        ocr_llm_executor,
                        reader.readtext,
                        image_from_page_np
                    )
                    ocr_tasks.append(ocr_task_future)
                    all_extracted_text_segments.append(f"--- Page {page_num + 1} (Full Page OCR Scheduled) ---\n")
                except Exception as e_page_ocr:
                    print(f"ERROR: Could not render or prepare page {page_num} for full page OCR: {e_page_ocr}")
            else:
                print(f"DEBUG: Skipping full page OCR for page {page_num} based on content heuristics.")

        doc.close() # Close the PDF document after processing all pages.

        print("Waiting for all OCR tasks to complete...")
        completed_ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        print("All OCR tasks completed.")

        for res in completed_ocr_results:
            if isinstance(res, Exception):
                print(f"ERROR: An OCR task failed: {res}")
            else:
                extracted_text_from_ocr = [text for (bbox, text, prob) in res]
                if extracted_text_from_ocr:
                    combined_ocr_text_segment = "\n".join(extracted_text_from_ocr)
                    all_extracted_text_segments.append(combined_ocr_text_segment)

    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        print(f"Processing {original_filename} as DOCX (direct text extraction)...")
        all_extracted_text_segments.append(_extract_text_from_docx(file_bytes))
        print("Text extracted from DOCX.")

    elif content_type == "application/msword":
        print(f"Processing {original_filename} as DOC (direct text extraction)...")
        all_extracted_text_segments.append(_extract_text_from_doc(file_bytes))
        print("Text extracted from DOC.")

    else:
        raise ValueError(f"Unsupported content type for processing: {content_type}")

    full_resume_text = "\n".join(all_extracted_text_segments)

    print("Calling LLM for full resume parsing...")
    parsed_resume_data = await _call_llm_for_resume_parsing(full_resume_text)
    print(f"LLM parsed data: {json.dumps(parsed_resume_data, indent=2)}")

    extracted_name = parsed_resume_data.get("basic_details", {}).get("name", "").strip()
    if extracted_name and extracted_name != "N/A":
        sanitized_name = re.sub(r'[^\w\s-]', '', extracted_name).strip().replace(' ', '_')
        output_filename = f"{sanitized_name}_resume.json"
    else:
        output_filename = f"parsed_resume.json"

    return {
        "filename": output_filename,
        "parsed_resume_data": parsed_resume_data
    }
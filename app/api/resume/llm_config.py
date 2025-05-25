import os
import logging
from openai import AzureOpenAI

logger = logging.getLogger(__name__) # Get a logger instance for this module

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize azure_openai_client. It will be None if credentials are not fully set.
azure_openai_client = None
if all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION]):
    try:
        azure_openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        logger.info("Azure OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}", exc_info=True)
        azure_openai_client = None
else:
    logger.warning("Azure OpenAI environment variables are not fully set. LLM functionality will be disabled.")


# Define the LLM prompt here.
LLM_RESUME_PARSING_PROMPT = """You are a Resume Data Extractor. When given a raw resume (plain text, PDF-OCR, or DOCX), you must output **only** a JSON object with the following schema:

{
  "basic_details": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "links": {
      "github": "string",
      "linkedin": "string"
      // any other personal URLs; if none, omit or set to "N/A"
    }
  },
  "technical_expertise": [
    "string"
    // e.g. ".NET Core", "React", "Azure"; gather all listed skills
  ],
  "certifications": [
    {
      "title": "string",
      "date": "string" // if no date is given, set to "N/A"
    }
  ],
  "professional_summary": "string", // A concise, high-level overview or narrative paragraph(s) of the candidate's career, key qualifications, and objectives. This may also be presented as high-level bullet points summarizing their overall profile. Look for sections like "Summary", "Professional Summary", "Profile", "About Me", etc. This field should NOT contain granular technical skills (which belong in 'technical_expertise') nor detailed job responsibilities (which belong in 'professional_experience'). If absent, "N/A".
  "professional_experience": [
    {
      "company": "string",
      "date_range": "string", // e.g. "Oct 2021 – Dec 2024"; if unclear, "N/A"
      "role": "string",
      "client_engagement": "string", // the client or project name; if none, "N/A"
      "program": "string", // program or module name; if none, "N/A"
      "responsibilities": [
        "string" // each bullet or sentence as one element
      ]
    }
  ]
}

**Rules:**
1. **Field defaults:** If any field cannot be found, set its value to the string `"N/A"`.
2. **Technical expertise:** Normalize to an array of individual technologies/frameworks/tools.
3. **Certifications:** Extract title and date if possible; otherwise date → `"N/A"`.
4. **Sort** the `professional_experience` array in **descending** order of the number of responsibilities (i.e. experiences with more responsibilities come first).
5. **Strict JSON only:** Do not output any explanatory text or markdown—just the JSON.
6. **Professional Summary:** Extract the overarching summary/profile section(s). This may be titled "Summary", "Professional Summary", "Profile", "About Me", etc. It should be a high-level narrative or key bullet points about the candidate's career and objectives. If multiple such sections are present, combine them into a single string. Do not include granular technical skills (for 'technical_expertise') or specific job duties (for 'professional_experience') in this field.

"""
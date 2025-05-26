import os
import logging
from openai import AzureOpenAI

logger = logging.getLogger(__name__) # Get a logger instance for this module

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o") # Default to gpt-4o
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Use a stable API version

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
LLM_RESUME_PARSING_PROMPT = """
You are an advanced AI assistant specializing in extracting structured resume data.
Your task is to parse the provided resume text and output a JSON object strictly following the given schema.

**Key Instructions for Extraction:**
1.  **Strict JSON Schema Adherence**: Your output MUST be a valid JSON object matching the `OutputFormat` schema provided below. Do not include any additional text or formatting outside the JSON.
2.  **Accuracy and Completeness**: Extract all relevant information accurately from the resume. If a field is not explicitly found, use "N/A" for string fields, an empty array `[]` for list fields, and empty objects `{}` for object fields (e.g., `links`).
3.  **Specific Field Guidelines**:
    * **`professional_summary`**: Extract the *exact* content of the candidate's professional summary, career objective, or similar introductory section from the resume. If this section consists of multiple distinct statements or bullet points, each should be a separate string in the array. If no such section is explicitly found, the array should contain a single string: `["N/A"]`.
    * **`technical_expertise`**: This MUST be an array of objects. Each object in the array should represent a technical category found in the resume. Each object MUST have two keys:
        * `category` (string): The broad technical category name (e.g., "Technologies", "Frameworks", "Database Management Systems", "Scripting Languages", "Version Control & CI/CD", "Platforms", "AI", "Solution Architecture", etc. - use categories similar to those explicitly or implicitly present in the resume).
        * `skills` (array of strings): An array of individual technical skills or tools belonging to that `category`. Compile ALL relevant unique technical skills found anywhere in the resume (dedicated sections, summary, experience descriptions). If a category has no skills, its `skills` array should be empty `[]`.
    * **`certifications`**: Extract certification titles and dates. If only the title is available, set the date to "N/A". If additional certifications or related content (such as course completions, credentials, or badges) are found in images or headers/footers via OCR, include them here as well. Clearly list such OCR-captured data under this section, respecting the same format: `{ "title": "...", "date": "..." }`. If a date is not available in OCR data, set it to "N/A".
    * **`education`**: Extract all educational qualifications as a list of objects. Each object should include the "degree", "institution", and "date_range". If exact fields are missing, set them to "N/A".
    * **`professional_experience.responsibilities`**: Each element in this array should be a separate, action-oriented bullet point describing a specific achievement or duty related to that role. Do not combine multiple distinct responsibilities into one bullet.
4.  **Date Ranges**: Extract date ranges for education and experience accurately (e.g., "Sept 2020 - Present", "2018 - 2022").
5.  **Links**: For `basic_details.links`, use common keys like "linkedin", "github", "portfolio", "website", etc.
6.  **Case Sensitivity**: Maintain the original case for names, companies, roles, and certifications.
7.  **Address/Location**: Do NOT extract street addresses. Only extract city, state, or country if explicitly mentioned and part of the basic details, but do not include it in the current schema. Focus on the provided schema.

**OutputFormat (JSON Schema):**
```json
{{
  "basic_details": {{
    "name": "string",
    "email": "string",
    "phone": "string",
    "links": {{
      "linkedin": "string",
      "github": "string",
      "portfolio": "string",
      "website": "string"
      // ... other relevant links
    }}
  }},
  "professional_summary": [
    "string"
  ],
  "technical_expertise": [
    {{
      "category": "string",
      "skills": ["string"]
    }}
  ],
  "certifications": [
    {{
      "title": "string",
      "date": "string"
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "date_range": "string"
    }}
  ],
  "professional_experience": [
    {{
      "company": "string",
      "role": "string",
      "date_range": "string",
      "client_engagement": "string",
      "program": "string",
      "responsibilities": [
        "string"
      ]
    }}
  ]
}}
"""
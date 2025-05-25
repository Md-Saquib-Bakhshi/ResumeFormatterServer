import os
from openai import AzureOpenAI

# --- Azure OpenAI Configuration ---
# Fetch credentials from environment variables.
# It's best practice to load these from environment variables for security and flexibility.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Validate that all necessary environment variables are set.
# This check should ideally happen at application startup (e.g., in main.py's lifespan event)
# to prevent the app from starting if credentials are missing.
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION]):
    # Raising an error here prevents the app from running with misconfigured LLM
    raise ValueError(
        "Missing Azure OpenAI environment variables. Please set "
        "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME, "
        "and AZURE_OPENAI_API_VERSION."
    )

# Initialize the Azure OpenAI client globally.
# This client will be reused across all LLM calls to avoid re-initialization overhead.
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Define the LLM prompt here. This keeps the prompt content centralized and easily modifiable.
# Using a triple-quoted string for multi-line clarity.
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
  "professional_summary": "string", // summary or profile paragraph; if absent, "N/A"
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
5. **Strict JSON only:** Do not output any explanatory text or markdown—just the JSON."""
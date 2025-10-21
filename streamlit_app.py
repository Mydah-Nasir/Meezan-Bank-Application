import streamlit as st
import cv2
# import pytesseract
# from pytesseract import Output
from PIL import Image
import openai
import base64
# import ollama
import tempfile
import json
import io
import os
from openai import OpenAI
import re
import numpy as np
import pandas as pd
import cv2
import requests
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Frame, Paragraph
from reportlab.lib.units import inch
import os
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
# from paddleocr import PaddleOCR
import hashlib
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.set_page_config(page_title="Banking AI Tools", layout="wide")

# --- Helper Functions ---
def file_hash(file_obj):
    return hashlib.md5(file_obj.getbuffer()).hexdigest()
def display_dict(data):
    for key, value in data.items():
        if isinstance(value, dict):
            st.subheader(key)
            #print(" " * indent + f"{key}:")  # heading
            display_dict(value)  # recursive call with indentation
        else:
            st.write(f"{key}: {value}")
            #print(" " * indent + f"{key}: {value}")
# ---- Parse Markdown into dict ----
def parse_response(text):
    data = {}

    # Match patterns like **Field:** value
    matches = re.findall(r"\*\*(.+?)\*\*:\s*([^\n]+)", text)
    for field, value in matches:
        data[field.strip()] = value.strip()

    return data



def render_dict(d, prefix=""):
    """Recursively render dictionary fields as editable Streamlit inputs"""
    updated = {}
    for key, value in d.items():
        field_key = f"{prefix}{key}"
        if isinstance(value, dict):
            st.subheader(key)
            updated[key] = render_dict(value, prefix=field_key + ".")
        else:
            # Choose appropriate input type
            if value in ["Single", "Married", "Yes", "No"]:
                options = ["Single", "Married"] if key.lower() == "marital status" else ["Yes", "No"]
                updated[key] = st.selectbox(key, options, index=options.index(value))
            else:
                updated[key] = st.text_input(key, value=str(value))
    return updated

def find_heading_y(image_gray, heading_text):
    ocr_data = pytesseract.image_to_data(image_gray, output_type=Output.DICT)
    for i, word in enumerate(ocr_data['text']):
        if heading_text.lower() in word.lower():
            return ocr_data['top'][i]
    return None

def save_to_pdf(data_dict1, data_dict2, data_dict3):
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "bank_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 60, "Meezan Bank Account Opening Extracted Data")
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_HEIGHT - 90, f"Day: {data_dict1.get('Day')} Month: {data_dict1.get('Month')} Year: {data_dict1.get('Year')}")
    c.drawString(2 * PAGE_WIDTH / 3, PAGE_HEIGHT - 90, f"Type of Account: {data_dict1.get('Type of Account')}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, PAGE_HEIGHT - 110, f"PRINICPAL ACCOUNT HOLDER")
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_HEIGHT - 130, f"Name MR./Mrs./Ms: {data_dict1.get('Name')}")
    c.drawString(40, PAGE_HEIGHT - 150, f"Father / Husband's Name: {data_dict1.get('Father/Husband Name')}")
    y = PAGE_HEIGHT - 150
    c.drawString(PAGE_WIDTH / 2, y, f"Mother's Maiden Name: {data_dict1.get('Mother Maiden Name')}")
    y-=20
    c.drawString(40, y, f"CNIC/NICOP/Passport No.: {data_dict1.get('CNIC/NICOP/Passport No')}")
    y-=20
    c.drawString(40, y, f"Issuance Date: {data_dict1.get('Issuance Date')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Expiry Date: {data_dict1.get('Expiry Date')}")
    y-=20
    c.drawString(40, y, f"Date of Birth: {data_dict1.get('Date of Birth')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Marital Status: {data_dict1.get('Marital Status')}")
    y-=20
    c.drawString(40, y, f"Religion: {data_dict1.get('Religion')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Place of Birth: {data_dict1.get('Place of Birth')}")
    y-=20
    c.drawString(40, y, f"Nationality: {data_dict1.get('Nationality')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Dual Nationality: {data_dict1.get('Dual Nationality')}")
    y-=20
    c.drawString(40, y, f"Mailing Address:")
    y-=20
    c.drawString(60, y, f"Street: {data_dict1.get('Mailing Address: Street')}")
    y-=20
    c.drawString(60, y, f"City: {data_dict1.get('Mailing Address: City')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Country: {data_dict1.get('Mailing Address: Country')}")
    y-=20
    c.drawString(40, y, f"Current Address as per CNIC:")
    y-=20
    c.drawString(60, y, f"Street: {data_dict1.get('Current Address: Street')}")
    y-=20
    c.drawString(60, y, f"City: {data_dict1.get('Current Address: City')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Country: {data_dict1.get('Current Address: Country')}")
    y-=20
    c.drawString(40,y,f"Residential Status {data_dict2.get('Residential Status')}")
    y-=20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"CONTACT DETAILS")
    c.setFont("Helvetica", 10)
    y-=20
    c.drawString(40, y, f"Email: {data_dict2.get('Email')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Mobile Network: {data_dict2.get('Mobile Network')}")
    y-=20
    c.drawString(40, y, f"Mobile: {data_dict2.get('Mobile')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Tel Res/Office: {data_dict2.get('Tel/Res Office')}")
    y-=20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, f"IN CASE OF MINOR ACCOUNT:")
    y-=20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Name of Guardian: {data_dict2.get('In Case of Minor Account: Name of Guardian')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Realtionship with Principal: {data_dict2.get('In Case of Minor Account: Relation with Principal')}")
    y-=20
    c.drawString(40, y, f"Guardian CNIC: {data_dict2.get('In Case of Minor Account: Guardian CNIC')}")
    c.drawString(PAGE_WIDTH / 2, y, f"CNIC Expiry Date : {data_dict2.get('In Case of Minor Account: CNIC Expiry Date')}")
    y-=20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, f"BANK ACCOUNT DETAIL OF PRINICPAL ACCOUNT HOLDER FOR REDEMPTION AND DIVIDEND PAYMENTS")
    c.setFont("Helvetica", 10)
    y-=20
    c.drawString(40, y, f"Bank Account No.(IBAN preferred): {data_dict2.get('Bank Account Detail: Bank Account No.')}")
    y-=20
    c.drawString(40, y, f"Bank Name: {data_dict2.get('Bank Account Detail: Bank')}")
    c.drawString(PAGE_WIDTH / 3, y, f" Branch: {data_dict2.get('Bank Account Detai: Branch')}")
    c.drawString(2 * PAGE_WIDTH / 3, y, f" City: {data_dict2.get('Bank Account Detail: City')}")
    y-=20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"JOINT ACCOUNT HOLDERS")
    c.setFont("Helvetica-Bold", 10)
    y-=20
    c.drawString(40, y, f"Joint Holder 1: ")
    y-=20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Realtion with Principal: {data_dict2.get('Joint Account Holders: Joint Holder 1: Relation with Principal')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Customer ID (if any) {data_dict2.get('Joint Account Holders: Joint Holder 1: Customer ID')}")
    y-=20
    c.drawString(40, y, f"Name: {data_dict2.get('Joint Account Holders: Joint Holder 1: Name')}")
    y-=20
    c.drawString(40, y, f"CNIC/NICOP/Passport No.: {data_dict2.get('Joint Account Holders: Joint Holder 1: CNIC/NICOP/Passport')}")
    y-=20
    c.drawString(40, y, f"Issuance Date: {data_dict2.get('Joint Account Holders: Joint Holder 1: Issuance Date')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Expiry Date: {data_dict2.get('Joint Account Holders: Joint Holder 1: Expiry Date')}")
    y-=20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, f"Joint Holder 2:")
    y-=20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Realtion with Principal: {data_dict2.get('Joint Account Holders: Joint Holder 2: Relation with Principal')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Customer ID (if any): {data_dict2.get('Joint Account Holders: Joint Holder 2: Customer ID')}")
    y-=20
    c.drawString(40, y, f"Name: {data_dict2.get('Joint Account Holders: Joint Holder 2: Name')}")
    y-=20
    c.drawString(40, y, f"CNIC/NICOP/Passport No.: {data_dict2.get('Joint Account Holders: Joint Holder 2: CNIC/NICOP/Passport')}")
    y-=20
    c.drawString(40, y, f"Issuance Date: {data_dict2.get('Joint Account Holders: Joint Holder 2: Issuance Date')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Expiry Date: {data_dict2.get('Joint Account Holders: Joint Holder 2: Expiry Date')}")
    c.showPage()
    y = PAGE_HEIGHT - 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"SPECIAL INSTRUCTIONS")
    y-=20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Account Operating Instructions: {data_dict3.get('Account Operating Instructions')}")
    y-=20
    c.drawString(40, y, f"Dividend Mandate: {data_dict3.get('Dividend Mandate')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Stock Dividend: {data_dict3.get('Stock Dividend')}")
    y-=20
    c.drawString(40, y, f"Communication Mode: {data_dict3.get('Communication Mode')}")
    y-=20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, f"DETAIL ABOUT MEEZAN TAHAFFUZ PENSION FUND (MTPF Account)")
    y-=20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Expected Retirement Date: {data_dict3.get('Expected Retirement Date')}")
    y-=20
    c.drawString(40, y, f"Allocation Scheme: {data_dict3.get('Allocation Scheme')}")
    c.save()
    return pdf_file

def call_ollama_api_with_image(image_file, prompt, model="llama3.2-vision"):
    """Call Ollama REST API with image input and vision-capable model"""

    try:
        # Read image and convert to bytes
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # Call ollama chat
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_bytes]  # pass bytes directly
            }]
        )
        print(response)

        return response.get("message", {}).get("content", "")
    
    except Exception as e:
        st.error(f"Error using Ollama chat API: {str(e)}")
        return None

def segment_image(image_pil):
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    width, height = image_pil.size

    y_contact = find_heading_y(image_cv, "CONTACT DETAILS") or int(height * 0.4)
    y_special = find_heading_y(image_cv, "SPECIAL INSTRUCTIONS") or int(height * 0.65)

    boxes = [
        (0, 0, width, y_contact),
        (0, y_contact, width, y_special),
        (0, y_special, width, height)
    ]

    segment_paths = []
    for i, box in enumerate(boxes, 1):
        segment = image_pil.crop(box)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        segment.save(temp_file.name)
        segment_paths.append(temp_file.name)

    return segment_paths

import re
def extract_value(text, field):
    if not isinstance(text, str):
        return None
    patterns = [
        rf"\*\*{re.escape(field)}:\*\* (.+?)(\n|$)",
        rf"{re.escape(field)}: (.+?)(\n|$)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            if val.lower() in ["[not provided]", "not applicable", "[not applicable]"]:
                return None
            return val
    return None
def extract_address_block(response_text: str, addr_type: str) -> dict:
    """
    Extracts address components (Street, City, Country) for Mailing/Current Address.
    """
    address = {}
    # Find the block for this address type
    block_pattern = rf"(?:\*|\+)\s*{re.escape(addr_type)}:\s*(.*?)(?=\n\*|\Z)"
    block_match = re.search(block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if block_match:
        block_text = block_match.group(1)

        # Now extract individual components from inside the block
        for field in ["Street", "City", "Country"]:
            pattern = rf"(?:\*|\+)\s*{field}:\s*(.*)"
            match = re.search(pattern, block_text, re.IGNORECASE)
            if match:
                address[field] = match.group(1).strip()
    return address
def parse_account_opening_response(response_text: str) -> dict:
    """
    Parses structured account opening details from Ollama response text.

    Args:
        response_text (str): The response text from Ollama.

    Returns:
        dict: Extracted data in key-value form.
    """
    data = {}

    # 3. General fields
    fields = [
        "Day","Month","Year","Type of Account","Name", "Father/Husband Name", "Mother Maiden Name", "CNIC/NICOP/Passport No",
        "Issuance Date", "Expiry Date", "Date of Birth", "Marital Status",
        "Religion", "Place of Birth", "Nationality", "Dual Nationality",
    ]
    for field in fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value

    # 4. Mailing and Current Address
    for addr_type in ["Mailing Address", "Current Address"]:
        address = extract_address_block(response_text, addr_type)
        if address:
            data[addr_type] = address   

    return data


def parse_account_info(response):
    parsed = {}

    parsed = {
        "Residential Status": extract_value(response, "Residential Status"),
        "Email": extract_value(response, "Email"),
        "Mobile Network": extract_value(response, "Mobile Network"),
        "Tel/Res Office": extract_value(response, "Tel/Res Office"),
        "Mobile": extract_value(response, "Mobile")
    }

    parsed["In Case of Minor Account"] = {
        "Name of Guardian": extract_value(response, "Name of Guardian"),
        "Relation with Principal": extract_value(response, "Relation with Principal"),
        "Guardian CNIC": extract_value(response, "Guardian CNIC"),
        "CNIC Expiry Date": extract_value(response, "CNIC Expiry Date")
    }

    parsed["Bank Account Detail"] = {
        "Bank Account No.": extract_value(response, "Bank Account No."),
        "Bank": extract_value(response, "Bank"),
        "Branch": extract_value(response, "Branch"),
        "City": extract_value(response, "City")
    }

    # Extract Joint Account Holder blocks
    joint_holders = {}
    matches = re.split(r"\*{0,2}Joint Holder \d:?\*{0,2}", response)
    if len(matches) > 1:
        # The first item is text before Joint Holder 1, skip it
        holder_texts = matches[1:]
        for idx, block in enumerate(holder_texts, 1):
            joint_holders[f"Joint Holder {idx}"] = {
                "Name": extract_value(block, "Name"),
                "Relation with Principal": extract_value(block, "Relation with Principal"),
                "Customer ID": extract_value(block, "Customer ID"),
                "CNIC/NICOP/Passport": extract_value(block, "CNIC/NICOP/Passport"),
                "Issuance Date": extract_value(block, "Issuance Date"),
                "Expiry Date": extract_value(block, "Expiry Date")
            }
    parsed["Joint Account Holders"] = joint_holders

    return parsed
def parse_special_instructions(response):
    parsed = {
            "Account Operating Instructions": extract_value(response, "Account Operating Instructions"),
            "Dividend Mandate": extract_value(response, "Dividend Mandate"),
            "Communication Mode": extract_value(response, "Communication Mode"),
            "Stock Dividend": extract_value(response, "Stock Dividend"),
            "Expected Retirement Date": extract_value(response, "Expected Retirement Date"),
            "Allocation Scheme": []
    }

    # Handle multiple Allocation Scheme options
    allocation_section = re.search(r"(?i)Alloca?tion Scheme[:\n]*((?:\*|\-).+?)(\n\n|\Z)", response, re.DOTALL)
    if allocation_section:
        allocations = re.findall(r"[\*\-]\s*(.+)", allocation_section.group(1))
        parsed["Allocation Scheme"] = [a.strip() for a in allocations]

    return parsed


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_prompt_for_segment(seg_id):
    prompts = {
        1: """Extract this structure only:
{
  "Date": {"Day": "", "Month": "", "Year": ""},
  "Type of Account": "",
  "Principal Account Holder": {
    "Name": "", "Father's/Husband's Name": "", "Mother's Maiden Name": "",
    "CNIC/NICOP/Passport No": "", "Issuance Date": "", "Expiry Date": "",
    "Date of Birth": "", "Marital Status": "", "Religion": "", "Place of Birth": "",
    "Nationality": "", "Dual Nationality": "",
    "Mailing Address": {"Street": "", "City": "", "Country": ""},
    "Current Address": {"Street": "", "City": "", "Country": ""}
  }
}""",
        2: """Extract this structure only:
{
  "Residential Status": "",
  "Contact Details": {
    "Email": "", "Mobile Network": "", "Tel/Res Office": "", "Mobile": ""
  },
  "In Case of Minor Account": {
    "Name of Guardian": "", "Relation with Principal": "", "Guardian CNIC": "", "CNIC Expiry Date": ""
  },
  "Bank Account Detail": {
    "Bank Account No.": "", "Bank": "", "Branch": "", "City": ""
  },
  "Joint Account Holders": {
    "Joint Holder 1": {
      "Name": "", "Relation with Principal": "", "Customer ID": "", 
      "CNIC/NICOP/Passport": "", "Issuance Date": "", "Expiry Date": ""
    },
    "Joint Holder 2": {
      "Name": "", "Relation with Principal": "", "Customer ID": "", 
      "CNIC/NICOP/Passport": "", "Issuance Date": "", "Expiry Date": ""
    }
  }
}""",
        3: """In allocation scheme add all that are ticked/checked. Leave blank if none is ticked. Extract this structure only:
{
  "Special Instructions": {
    "Account Operating Instructions": "", "Dividend Mandate": "", "Communication Mode": "", "Stock Dividend": ""
  },
  "Detail About Meezan Tahaffuz Pension Fund (MTPF) Account": {
    "Expected Retirement Date": "", "Allocation Scheme": [ ]
  }
}"""
    }
    return prompts.get(seg_id, "")

def answer_question_with_pdf(model_name, pdf_path):
    context = extract_text_from_pdf(pdf_path)
    model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model = GPT4All(model_name)

    prompt = f"""Extract the following fields from the document below and return the result strictly in this JSON format:

    {{
      "Date": {{"Day": "", "Month": "", "Year": ""}},
      "Type of Account": "",
      "Principal Account Holder": {{
        "Name": "", "Father's/Husband's Name": "", "Mother's Maiden Name": "",
        "CNIC/NICOP/Passport No": "", "Issuance Date": "", "Expiry Date": "",
        "Date of Birth": "", "Marital Status": "", "Religion": "", "Place of Birth": "",
        "Nationality": "", "Dual Nationality": "",
        "Mailing Address": {{"Street": "", "City": "", "Country": ""}},
        "Current Address": {{"Street": "", "City": "", "Country": ""}}
      }}
    }}

    Context:
    {context}

    Return only the JSON. Do not explain anything else.
    """

    answer = model.generate(prompt)

def call_gpt_vision(image_path, seg_id, api_key):
    base64_image = encode_image_base64(image_path)
    prompt = get_prompt_for_segment(seg_id)

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You extract structured data from forms."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000
    )

    try:
        content = response.choices[0].message.content.strip()
        print(content)
        # Remove the backticks and "json" label
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            cleaned_json = match.group(1).strip()
        else:
            cleaned_json = content.strip()

        # Convert JSON string to dictionary
        return json.loads(cleaned_json)
    except Exception as e:
        return {"error": f"Failed to parse response: {e}", "raw": content}

OPENAI_API_KEY = st.secrets["API_KEY"]
api_key = OPENAI_API_KEY
def call_openai_api_with_image(image_file, prompt, model="gpt-4o"):
    """Call OpenAI GPT-4o API with Streamlit-uploaded image and text prompt."""

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Open and re-encode uploaded image file to base64
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")  # ensure JPEG format
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Compose message content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]

        # Call OpenAI GPT-4o chat completion endpoint
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        # Extract text reply
        reply = response.choices[0].message.content
        return reply

    except Exception as e:
        st.error(f"Error using OpenAI GPT-4o API: {str(e)}")
        return None

def display_flat_dict(data, prefix=""):
    if isinstance(data, dict):
        for key, value in data.items():
            display_flat_dict(value, f"{prefix}{key} > " if prefix else f"{key}: ")
    else:
        st.markdown(f"**{prefix.strip(' >')}:** {data}")
def flatten_dict(d, parent_key='', sep=': '):
    """Flatten nested JSON"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)
# --- Streamlit App UI ---

#st.sidebar.title("⚙️ Configuration")
#OPENAI_API_KEY = st.secrets["API_KEY"]

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["Account Opening Form Extractor", "Credit Score Predictor"])
if page == "Account Opening Form Extractor":
    st.title("📄 Bank Account Opening Form for Individual AI Extractor")
    uploaded_file = st.file_uploader("📤 Upload scanned form", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        new_hash = file_hash(uploaded_file)
        # Check if file changed
        if "file_hash" not in st.session_state or st.session_state.file_hash != new_hash:
            # Reset session_state for new file
            print('New File')
            st.session_state.file_hash = new_hash
            for key in ["response", "response2", "response3"]:
                if key in st.session_state:
                    del st.session_state[key]
        print('re-runned')
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Form", use_container_width=True)
        prompt = """Extract the following details from the form in a structured and complete manner:
    * Date: [Your answer here]
    * Day: [Your answer here]
    * Month: [Your answer here]
    * Year: [Your answer here]
    * Type of Account: [Your answer here]
    * Principal Account Holder:
        * Name: [Your answer here]
        * Father/Husband Name: [Your answer here]
        * Mother Maiden Name: [Your answer here]
        * CNIC/NICOP/Passport No: [Your answer here]
        * Issuance Date: [Your answer here]
        * Expiry Date: [Your answer here]
        * Date of Birth: [Your answer here]
        * Marital Status: [Your answer here]
        * Religion: [Your answer here]
        * Place of Birth: [Your answer here]
        * Nationality: [Your answer here]
        * Dual Nationality: [Your answer here]
    * Mailing Address:
        * Street: [Your answer here]
        * City: [Your answer here]
        * Country: [Your answer here]
    * Current Address:
        * Street: [Your answer here]
        * City: [Your answer here]
        * Country: [Your answer here]

    Leave any field blank if the information is missing or not available."""

        #response = call_ollama_api_with_image(uploaded_file, prompt)
        if "response" not in st.session_state:
            response = call_openai_api_with_image(uploaded_file, prompt)
            st.session_state.response = response   
        prompt2 = "Extract the following details from the form in a structured and complete manner: * Residential Status: [Your answer here] * Email: [Your answer here] * Mobile Network: [Your answer here] * Tel/Res Office: [Your answer here] * Mobile: [Your answer here] * In Case of Minor Account: * Name of Guardian: [Your answer here] * Relation with Principal: [Your answer here] * Guardian CNIC: [Your answer here] * CNIC Expiry Date: [Your answer here] * Bank Account Detail: * Bank Account No.: [Your answer here] * Bank: [Your answer here] * Branch: [Your answer here] * City: [Your answer here] * Joint Account Holders: * Joint Holder 1: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here] * Joint Holder 2: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here]Leave blank if missing"
        #response2 = call_ollama_api_with_image(uploaded_file, prompt2)
        if "response2" not in st.session_state:
            response2 = call_openai_api_with_image(uploaded_file, prompt2)
            st.session_state.response2 =  response2
        prompt3 = """Extract the following details from the form should in a structured and complete manner:

    Special Instructions:
    Account Operating Instructions: [e.g., Either or Survivor, Jointly Operated, etc.]

    Dividend Mandate: [e.g., Credit to Bank Account, Reinvest, etc.]
    Communication Mode: [e.g., Email, Postal Mail, etc.]

    Stock Dividend: [e.g., Yes, No, or method of delivery]

    Detail About Meezan Tahaffuz Pension Fund (MTPF) Account:
    Expected Retirement Date: [DD/MM/YYYY]

    Allocation Scheme: [List all ticked or checked options, e.g., Equity, Debt, Money Market. Leave blank if none are selected.]

    Please ensure only the checked/ticked values are included in the "Allocation Scheme" list. If no boxes are selected, return an empty list."""
        #response3 = call_ollama_api_with_image(uploaded_file, prompt3)
        if "response3" not in st.session_state:
            response3 = call_openai_api_with_image(uploaded_file, prompt3)
            st.session_state.response3 = response3
        # print(response)
        if "response" in st.session_state:
            parsed_data = parse_account_opening_response(st.session_state.response)
            # print('parsed data:',parsed_data)
            flat_data = flatten_dict(parsed_data)
            parsed_data2 = parse_account_info(st.session_state.response2)
            flat_data2 = flatten_dict(parsed_data2)
            parsed_data3 = parse_special_instructions(st.session_state.response3)
            flat_data3 = flatten_dict(parsed_data3)
            # form_data = parse_response(response)

            st.write("### Edited Form Data")

        # ---- Display Editable Form ----
            with st.form("bank_form"):
                edited_form = {}
                edited_form2= {}
                edited_form3 = {}
                # Create two columns
                cols = st.columns(2)
                # Loop with index to alternate between columns
                for idx, (field, value) in enumerate(flat_data.items()):
                    col = cols[idx % 2]  # Alternate between col[0] and col[1]
                    with col:
                        edited_form[field] = st.text_input(field, value)
                for idx, (field, value) in enumerate(flat_data2.items()):
                    col = cols[idx % 2]  # Alternate between col[0] and col[1]
                    with col:
                        edited_form2[field] = st.text_input(field, value)
                for idx, (field, value) in enumerate(flat_data3.items()):
                    col = cols[idx % 2]  # Alternate between col[0] and col[1]
                    with col:
                        edited_form3[field] = st.text_input(field, value)
                submitted = st.form_submit_button("Save Form")
            # print(response2)
            # print(response3)
            # st.write("### Response:")
            st.subheader('Principal Account Holder')
            st.write(st.session_state.response)
            # print('edited form', edited_form)
            st.subheader('Contact Details')
            parsed_data2 = parse_account_info(st.session_state.response2)
            # print('parsed data2:',parsed_data2)
            #flat_data2 =flatten_dict(parsed_data2)
            display_dict(parsed_data2)
            parsed_data3 = parse_special_instructions(st.session_state.response3)
            st.subheader('Special Instructions')
            # print('parsed data3:',parsed_data3)
            display_dict(parsed_data3)
            st.write(st.session_state.response2)
            st.write(st.session_state.response3)
            # combined_data = {
            #         "First Response (Editable)": editable_response,
            #         "Second Response": response2,
            #         "Third Response": response3
            #     }
            print('Form2',edited_form2)
            print('form3',edited_form3)
            pdf_path = save_to_pdf(edited_form, edited_form2, edited_form3)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="bank_form_data.pdf", mime="application/pdf")
        # parsed_data = parse_account_opening_response(response)
        # parsed_data2 = parse_account_info(response2)
        # parsed_data3 = parse_special_instructions(response3)
        # combined_data = {**parsed_data, **parsed_data2}
        # Flatten the combined dictionary
        # flat_data = flatten_dict(parsed_data)
        # flat_data2 =flatten_dict(parsed_data2)
        # flat_data3 = flatten_dict(parsed_data3)

        # Create a DataFrame
        # df = pd.DataFrame(list(flat_data.items()), columns=["Field", "Value"])
        # df2 = pd.DataFrame(list(flat_data2.items()), columns=["Field", "Value"])
        # df3 = pd.DataFrame(list(flat_data3.items()), columns=["Field", "Value"])

        # Display in Streamlit
        # st.dataframe(df)
        # st.dataframe(df2)
        # st.dataframe(df3)

        # with st.spinner("Processing image..."):
        #     segments = segment_image(image_pil)
            # pdf_paths = apply_ocr_and_generate_pdfs(segments)
            # print("Generated PDF paths:", pdf_paths)

        # all_data = {}
        # for idx, seg_path in enumerate(segments, start=1):
        #     data = call_gpt_vision(seg_path, idx, OPENAI_API_KEY)

        #     if "error" in data:
        #         st.error(data["error"])
        #         st.text(data["raw"])
        #     else:
        #         all_data.update(data)

        # st.success("✅ Bank Form Processed.")
        # flat_data = flatten_dict(all_data)
        # df = pd.DataFrame(list(flat_data.items()), columns=["Field", "Value"])
        # st.dataframe(df, use_container_width=True)

    # elif uploaded_file:
    #     st.warning("Please enter your credentials.")
elif page == "Credit Score Predictor":
    st.title("💳 Credit Score Predictor")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Estimate Your Credit Score")
        st.write("""
        Your credit score is one of the most important numbers in your financial life. 
        It can determine whether you can get financing and can mean the difference between 
        a preferential rate that saves you thousands of dollars or a more expensive loan.
        """)

    with col2:
        st.image("https://ecm.capitalone.com/WCM/creditwise/cw-simulator-banner-b4.d-desktop.png")
    # Form inputs
    st.header("Enter Your Information")
    st.write("Estimate your credit score in about 30 seconds. Just answer a few simple questions about your past credit usage:")

    with st.form("credit_score_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100)
            annual_income = st.number_input("Annual Income", min_value=0)
            num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0)
            num_credit_cards = st.number_input("Number of Credit Cards", min_value=0)
            interest_rate = st.number_input("Interest Rate", min_value=0)
            num_loans = st.number_input("Number of Loans", min_value=0)
            delay_days = st.number_input("Delay from due date", min_value=0)
            num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0)
            
        with col2:
            credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
            outstanding_debt = st.number_input("Outstanding Debt", min_value=0)
            credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0)
            credit_history_age = st.number_input("Credit History Age", min_value=0)
            monthly_emi = st.number_input("Total EMI per month", min_value=0)
            monthly_investment = st.number_input("Amount invested monthly", min_value=0)
            payment_behavior = st.number_input("Payment Behaviour", min_value=0)
            monthly_balance = st.number_input("Monthly Balance", min_value=0)
            payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])

        submitted = st.form_submit_button("Calculate Credit Score")

    if submitted:
        # Process credit mix
        credit_mix_map = {"Bad": 1, "Standard": 2, "Good": 3}
        credit_mix_value = credit_mix_map[credit_mix]
        
        # Process payment of min amount
        pma_nm = 1 if payment_min_amount == "NM" else 0
        pma_no = 1 if payment_min_amount == "No" else 0 
        pma_yes = 1 if payment_min_amount == "Yes" else 0

        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Annual_Income': [annual_income],
            'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_cards],
            'Interest_Rate': [interest_rate],
            'Num_of_Loan': [num_loans],
            'Delay_from_due_date': [delay_days],
            'Num_of_Delayed_Payment': [num_delayed_payments],
            'Credit_Mix': [credit_mix_value],
            'Outstanding_Debt': [outstanding_debt],
            'Credit_Utilization_Ratio': [credit_utilization],
            'Credit_History_Age': [credit_history_age],
            'Total_EMI_per_month': [monthly_emi],
            'Amount_invested_monthly': [monthly_investment],
            'Payment_Behaviour': [payment_behavior],
            'Monthly_Balance': [monthly_balance],
            'PMA_NM': [pma_nm],
            'PMA_No': [pma_no],
            'PMA_Yes': [pma_yes]
        })

        # Scale the features
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
        
        # Load model and make prediction
        model = xgb.Booster()
        model.load_model('model.h5')
        dmatrix = xgb.DMatrix(scaled_data)
        prediction = model.predict(dmatrix)

        # Show results
        st.header("Results")
        if prediction == 0:
            st.error("Credit Score: Poor")
            st.write("A poor credit score means that you are not eligible for application of loans")
        elif prediction == 1:
            st.warning("Credit Score: Standard")
            st.write("A standard credit score means that you will likely be eligible for application of small amount for loan")
        else:
            st.success("Credit Score: Good")
            st.write("A good credit score means that you will likely be eligible for application of large sum for loan")

    # Additional information section
    st.markdown("""
    ## What Is a Credit Score?

    A credit score is tabulated by credit bureaus, who get information from the banks and companies you do business with about your financial payments. Your score is based on five basic things:

    * How often you pay your bills on time and how often you're late (35% of score)
    * How much you owe (10% of score)
    * How many types of debts and credit lines you have (10% of score)
    * Credit history length (15% of score)
    * Number of recent credit inquiries (10% of score)

    **Note:** Your income is not directly factored into your credit score.
    """)

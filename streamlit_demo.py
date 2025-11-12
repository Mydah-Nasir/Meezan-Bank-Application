import streamlit as st
import base64
from pathlib import Path
import streamlit as st
import cv2
import time
# import pytesseract
# from pytesseract import Output
from PIL import Image
import openai
import base64
#import ollama
import tempfile
import json
import io
import os
import subprocess
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
import os
import hashlib
# import torch

#----------Functions----------
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
import subprocess
def restart_ollama():
    try:
        subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=True)
        subprocess.run(["sudo", "systemctl", "start", "ollama"], check=True)
        #st.success("✅ Ollama service restarted successfully.")
        print("Ollama running successfully")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Failed to restart Ollama: {e}")


MODEL_PATH = "./llama_vision_model"  # local saved model folder

@st.cache_resource
def load_local_model():
    """Load and cache the locally saved Llama Vision model"""
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu"   # or "auto" if you have GPU
    )
    return processor, model


def call_local_model_with_image(image_file, prompt):
    """Run inference using the locally saved model"""
    try:
        processor, model = load_local_model()

        # Open image
        image = Image.open(image_file).convert("RGB")

        # Preprocess input
        inputs = processor(images=image, text=prompt, return_tensors="pt")

        # Move tensors to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)

        # Decode output
        result = processor.decode(outputs[0], skip_special_tokens=True)
        print(result)
        return result

    except Exception as e:
        st.error(f"Error running local model: {str(e)}")
        return None
def ensure_ollama_running():
    """Start Ollama server if it's not already running."""
    try:
        # Test connection to Ollama server
        requests.get("http://localhost:11434/ping", timeout=2)
    except:
        # Start Ollama server in background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)




# def call_ollama_api_with_image(image_file, prompt, model="qwen2.5vl:7b"):
#     """Call Ollama client with image and forcefully reset connection after each use."""
#     #ensure_ollama_running()
#     try:
#         # Convert image to bytes
#         image = Image.open(image_file)
#         buffered = io.BytesIO()
#         image.save(buffered, format="JPEG")
#         image_bytes = buffered.getvalue()

#         # Initialize new client (default connects to localhost:11434)
#         client = ollama.Client()

#         # Call Ollama chat
#         response = client.chat(
#             model=model,
#             messages=[{
#                 "role": "user",
#                 "content": prompt,
#                 "images": [image_bytes],
#             }],
#             stream=False
#         )
#         return response.get("message", {}).get("content", "")

#     except Exception as e:
#         st.error(f"Error using Ollama chat API: {str(e)}")
#         return None
OPENAI_API_KEY = st.secrets["API_KEY"]
api_key = OPENAI_API_KEY
def call_openai_api_with_image(image_file, prompt=None, model="gpt-4o"):
    """Call OpenAI GPT-4o API with Streamlit-uploaded image and text prompt."""

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Define a default extraction prompt
        default_prompt = (
            "You are an expert in OCR and document analysis. "
            "Carefully review the uploaded image and extract all visible information from it. "
            "Present the extracted content in a clear and structured format using readable sections and bullet points. "
            "Include key details such as document title, headers, names, dates, addresses, reference numbers, "
            "amounts, item descriptions, totals, and any tabular or listed data. "
            "If some information is unclear or partially visible, indicate it as '(unclear)'. "
            "Avoid adding extra commentary or assumptions — only include what is visible in the document."
        )


        # Use default prompt if user did not provide one
        effective_prompt = prompt.strip() if prompt else default_prompt

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
                    {"type": "text", "text": effective_prompt},
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

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Form AI Parser", layout="wide")

# ---------- BRAND COLORS ----------
PRIMARY_COLOR = "#4B006E"  # Meezan purple
SECONDARY_COLOR = "#007A4D"  # Meezan green
BACKGROUND_COLOR = "#F8F9FA"

# ---------- LOGO PATHS ----------
logo_path = "OfficeFlow Ai-01-01.png"       # your Meezan Bank logo
dashboard_preview = "dashboard.png"  # your dashboard screenshot

# ---------- CUSTOM CSS ----------
st.markdown(
    f"""
    <style>
    body {{
        background-color: {BACKGROUND_COLOR};
        font-family: 'Inter', sans-serif;
    }}
    .left-side {{
        background: linear-gradient(145deg, {PRIMARY_COLOR} 40%, {SECONDARY_COLOR} 100%);
        height: 100vh;
        width: 100%;
        color: white;
        padding: 3rem 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .left-side img {{
        width: 90%;
        max-width: 600px;
        border-radius: 20px;
        box-shadow: 0 0 25px rgba(0,0,0,0.2);
    }}
    .login-card {{
        background-color: white;
        padding: 3rem 3rem;
        border-radius: 20px;
        max-width: 420px;
        margin: auto;
        text-align: center;
    }}
    h2 {{
        color: {PRIMARY_COLOR};
        font-weight: 800;
    }}
    p {{
        
        margin-bottom: 2rem;
    }}
    .stTextInput > div > div > input {{
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
    }}
    .stButton > button {{
        width: 100%;
        border-radius: 10px;
        background-color: {PRIMARY_COLOR};
        color: white !important; 
        font-weight: 600;
        height: 3rem;
        border: none;
        font-color: white;
       
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
        color: white !important; 
    }}
    .forgot {{
        text-align: right;
        margin-bottom: 20px;
    }}
    .forgot a {{
        color: {SECONDARY_COLOR};
        font-size: 0.9rem;
        text-decoration: none;
    }}
    .forgot a:hover {{
        text-decoration: underline;
    }}
    .two-column-container {{
        display: flex;
        gap: 2rem;
        align-items: flex-start;
        justify-content: space-between;
        }}
    
    
    </style>
    """,
    unsafe_allow_html=True
)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if st.session_state.logged_in:
    st.title("📄 AI Form Extractor")
    st.sidebar.image('disruptlabs-logo-1.png',use_container_width='auto')
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))
    option = st.sidebar.selectbox(
    "Please select the appropriate form type:",
    ("Meezan Bank Form", "Custom Form"),
    )
    uploaded_file = st.sidebar.file_uploader("📤 Upload scanned form", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file:
        new_hash = file_hash(uploaded_file)
        # Check if file changed
        if "file_hash" not in st.session_state or st.session_state.file_hash != new_hash:
            # Reset session_state for new file
            print('New File')
            st.session_state.file_hash = new_hash
            # restart_ollama()
            for key in ["response", "response2", "response3"]:
                if key in st.session_state:
                    del st.session_state[key]
        print('re-runned')
        image_pil = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image_pil, caption="Uploaded Form", use_container_width=True)
        #st.image(image_pil, caption="Uploaded Form", use_container_width=True)
        if option == "Meezan Bank Form":
            prompt = """Extract the following details from the form in a structured and complete manner: * Date: [Your answer here] * Day: [Your answer here] * Month: [Your answer here] * Year: [Your answer here] * Type of Account: [Your answer here] * Principal Account Holder: * Name: [Your answer here] * Father/Husband Name: [Your answer here] * Mother Maiden Name: [Your answer here] * CNIC/NICOP/Passport No: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here] * Date of Birth: [Your answer here] * Marital Status: [Your answer here] * Religion: [Your answer here] * Place of Birth: [Your answer here] * Nationality: [Your answer here] * Dual Nationality: [Your answer here] * Mailing Address: * Street: [Your answer here] * City: [Your answer here] * Country: [Your answer here] * Current Address: * Street: [Your answer here] * City: [Your answer here] * Country: [Your answer here] Leave any field blank if the information is missing or not available."
    """

            if "response" not in st.session_state:
                response = call_openai_api_with_image(uploaded_file, prompt)
                st.session_state.response = response   
            prompt2 = "Extract the following details from the form in a structured and complete manner: * Residential Status: [Your answer here] * Email: [Your answer here] * Mobile Network: [Your answer here] * Tel/Res Office: [Your answer here] * Mobile: [Your answer here] * In Case of Minor Account: * Name of Guardian: [Your answer here] * Relation with Principal: [Your answer here] * Guardian CNIC: [Your answer here] * CNIC Expiry Date: [Your answer here] * Bank Account Detail: * Bank Account No.: [Your answer here] * Bank: [Your answer here] * Branch: [Your answer here] * City: [Your answer here] * Joint Account Holders: * Joint Holder 1: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here] * Joint Holder 2: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here]Leave blank if missing"
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

            # ---- Display Editable Form ----
                # Right column - Editable Form
                with col2:
                    st.markdown("### 📝 Edited Form Data")
                    with st.container(height=700):
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
                    st.sidebar.download_button("Download PDF", f, file_name="bank_form_data.pdf", mime="application/pdf", use_container_width=True)
        else:
            custom_response = call_openai_api_with_image(uploaded_file)
            with col2:
                st.subheader("Extracted Data from Custom Form")
                st.write(custom_response)

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

    #st.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))
# ---------- LAYOUT ----------
else:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        img_path = Path(dashboard_preview)  # keep your existing variable name
        if img_path.exists():
            img_bytes = img_path.read_bytes()
            mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            html = f'<div class="left-side"><img src="data:{mime};base64,{b64}" alt="dashboard preview"/></div>'
        else:
            html = f'<div class="left-side"><div style="color:#fff">Dashboard image not found: {dashboard_preview}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.image(logo_path, width=150)
        st.markdown("<h2>Officeflow AI</h2>", unsafe_allow_html=True)
        
        st.subheader("Login")
        st.write("Welcome! Please enter your credentials to proceed.")
        
        email = st.text_input("Email Address", placeholder="Enter your email")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        
        login = st.button("Login", type="primary")
        
        if login:
            if email == "admin@officeflowai.com" and password == "12345":
                st.session_state.logged_in = True
                st.rerun()  # reloads the app to show welcome screen
            else:
                st.error("Invalid credentials. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)

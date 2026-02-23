import streamlit as st
import base64
from pathlib import Path
import cv2
import time
from PIL import Image
import openai
import tempfile
import json
import io
import os
import subprocess
from openai import OpenAI
import re
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Frame, Paragraph
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
import hashlib
import requests
from PIL import Image
# import pytesseract
# from pytesseract import Output
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from google.oauth2 import service_account
from google.cloud import vision

# Load from secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp"]
)

vision_client = vision.ImageAnnotatorClient(credentials=credentials)

def find_heading_y(image_input, heading_text: str):
    """
    Detects the top Y coordinate of a heading using Google Cloud Vision OCR.
    Accepts either:
        - PIL Image
        - NumPy array (OpenCV)
    Returns: int Y-coordinate or None.
    """

    # ---------- HANDLE PIL OR NUMPY ----------
    if isinstance(image_input, np.ndarray):  
        # Convert NumPy → PIL
        image_pil = Image.fromarray(image_input)
    else:
        image_pil = image_input  # already PIL

    # ---------- PIL → JPEG bytes ----------
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    content = img_bytes.getvalue()

    # ---------- Google Vision detection ----------
    image = vision.Image(content=content)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    # ---------- Search each word for heading ----------
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join([symbol.text for symbol in word.symbols])

                    if heading_text.lower() in text.lower():
                        ys = [v.y for v in word.bounding_box.vertices]
                        return min(ys)

    return None


# def find_heading_y(image_gray, heading_text):
#     ocr_data = pytesseract.image_to_data(image_gray, output_type=Output.DICT)
#     for i, word in enumerate(ocr_data['text']):
#         if heading_text.lower() in word.lower():
#             return ocr_data['top'][i]
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
# Try to import PDF libraries
try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PyPDFium2 not available - PDF support limited")

#----------Functions----------
def convert_pdf_to_images(pdf_file):
    """Convert PDF file to list of PIL Images using pypdfium2"""
    try:
        # Save uploaded PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Convert PDF to images using pypdfium2
        pdf = pdfium.PdfDocument(tmp_path)
        images = []
        
        for page_number in range(len(pdf)):
            page = pdf.get_page(page_number)
            bitmap = page.render(scale=2.0)  # Higher scale for better quality
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        
        # Clean up
        pdf.close()
        os.unlink(tmp_path)
        
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return None

def file_hash(file_obj):
    return hashlib.md5(file_obj.getbuffer()).hexdigest()

def display_dict(data):
    for key, value in data.items():
        if isinstance(value, dict):
            st.subheader(key)
            display_dict(value)
        else:
            st.write(f"{key}: {value}")

def parse_response(text):
    data = {}
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
            if value in ["Single", "Married", "Male", "Female", "Transgender", "Yes", "No", "Filer", "Non-Filer"]:
                if key.lower() == "marital status":
                    options = ["Single", "Married"]
                elif key.lower() == "gender":
                    options = ["Male", "Female", "Transgender"]
                elif key.lower() == "zakat deduction":
                    options = ["Yes", "No"]
                elif key.lower() == "tax status":
                    options = ["Filer", "Non-Filer"]
                else:
                    options = ["Yes", "No"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "purpose":
                options = ["NEW ACCOUNT OPENING", "EXISTING ACCOUNT REG NO. REGULARIZATION", "EXISTING ACCOUNT REG NO. UPGRADATION"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "mailing address":
                options = ["RESIDENTIAL ADDRESS", "OFFICE/BUSINESS ADDRESS"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "statement delivery":
                options = ["By Email", "By Post"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "account type":
                options = ["Single", "Joint", "Minor", "Sole Proprietorship"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "resident status":
                options = ["Resident", "Non-Resident"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "type of institution":
                options = ["Individual", "Corporate", "Institution"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "dividend distribution option":
                options = ["Reinvest Dividend", "Credit to Bank Account", "Pay by Cheque"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "communication mode":
                options = ["Email", "Postal Mail", "SMS"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "account operating instructions":
                options = ["Either or Survivor", "Jointly Operated", "Anyone or Survivor"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "nature of business":
                options = ["Retail", "Trading", "Services", "Manufacturing", "Self-employed professional", "Other"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "business/employment details":
                options = ["Federal Government Employee", "Provincial Government Employee", "Semi-Government Employee", "Private Service", "Allied Bank Employee", "Other"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "education":
                options = ["Up to Matric/O-Level", "Intermediate/A-Level", "Bachelors"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "profession":
                options = ["Student", "Housewife", "Agriculturist", "Other"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            elif key.lower() == "occupation":
                options = ["Self-employed", "Salaried", "Other"]
                default_index = options.index(value) if value in options else 0
                updated[key] = st.selectbox(key, options, index=default_index)
            else:
                updated[key] = st.text_input(key, value=str(value))
    return updated

OPENAI_API_KEY = st.secrets["API_KEY"]
api_key = OPENAI_API_KEY

def call_openai_api_with_image(image_file, prompt=None, model="gpt-4o"):
    """Call OpenAI GPT-4o API with Streamlit-uploaded image/PDF and text prompt."""
    try:
        client = OpenAI(api_key=api_key)

        default_prompt = (
            "You are an expert in OCR and document analysis. "
            "Carefully review the uploaded image and extract all visible information from it. "
            "Present the extracted content in a clear and structured format using readable sections and bullet points. "
            "Include key details such as document title, headers, names, dates, addresses, reference numbers, "
            "amounts, item descriptions, totals, and any tabular or listed data. "
            "If some information is unclear or partially visible, indicate it as '(unclear)'. "
            "Avoid adding extra commentary or assumptions — only include what is visible in the document."
        )

        effective_prompt = prompt.strip() if prompt else default_prompt

        # Check if the file is a PDF
        if hasattr(image_file, 'type') and image_file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("PDF processing is not available. Please install pypdfium2.")
                return None
                
            # Convert PDF to images and use the first page
            pdf_images = convert_pdf_to_images(image_file)
            if pdf_images and len(pdf_images) > 0:
                image = pdf_images[0]  # Use first page
                st.info(f"📄 PDF detected. Processing first page of {len(pdf_images)} pages.")
            else:
                st.error("Failed to convert PDF to images.")
                return None
        else:
            # It's an image file
            image = Image.open(image_file)
        
        # Convert to RGB if necessary and prepare for API
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        reply = response.choices[0].message.content
        print(reply)
        return reply

    except Exception as e:
        st.error(f"Error using OpenAI GPT-4o API: {str(e)}")
        return None

def call_qwen_model_with_image(image_file, prompt=None, model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic"):
    """Call OpenAI Qwen with Streamlit-uploaded image/PDF and text prompt."""
    try:
        client = OpenAI( base_url="https://router.huggingface.co/v1",api_key=st.secrets["HF_TOKEN"],)

        default_prompt = (
            "You are an expert in OCR and document analysis. "
            "Carefully review the uploaded image and extract all visible information from it. "
            "Present the extracted content in a clear and structured format using readable sections and bullet points. "
            "Include key details such as document title, headers, names, dates, addresses, reference numbers, "
            "amounts, item descriptions, totals, and any tabular or listed data. "
            "If some information is unclear or partially visible, indicate it as '(unclear)'. "
            "Avoid adding extra commentary or assumptions — only include what is visible in the document."
        )

        effective_prompt = prompt.strip() if prompt else default_prompt

        # Check if the file is a PDF
        if hasattr(image_file, 'type') and image_file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("PDF processing is not available. Please install pypdfium2.")
                return None
                
            # Convert PDF to images and use the first page
            pdf_images = convert_pdf_to_images(image_file)
            if pdf_images and len(pdf_images) > 0:
                image = pdf_images[0]  # Use first page
                st.info(f"📄 PDF detected. Processing first page of {len(pdf_images)} pages.")
            else:
                st.error("Failed to convert PDF to images.")
                return None
        else:
            # It's an image file
            image = Image.open(image_file)
        
        # Convert to RGB if necessary and prepare for API
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = 0.0,
            top_p = 0.9

        )

        reply = response.choices[0].message.content
        print(reply)
        return reply

    except Exception as e:
        st.error(f"Error using OpenAI GPT-4o API: {str(e)}")
        return None

def extract_value(text, field):
    if not isinstance(text, str):
        return None
    patterns = [
        rf"\*\*{re.escape(field)}:\*\* (.+?)(\n|$)",
        rf"\*\s*\*\*\s*{re.escape(field)}\s*:\s*\*\*\s*([^\*\n\r]+)",
        rf"{re.escape(field)}: (.+?)(\n|$)",
        rf"{re.escape(field)}\s*-\s*(.+?)(\n|$)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            if val.lower() in ["[not provided]", "not applicable", "[not applicable]", "blank", "(blank)"]:
                return None
            return val
    return None

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

# ---------- MCB Redemption Forms Specific Functions ----------
def parse_mcb_redemption_c1(response_text: str) -> dict:
    """Parse MCB Redemption Request Form For Plans And Funds - C-1"""
    data = {}
    
    # Header Information
    header_fields = ["Date", "Investor Registration Number", "CNIC/NICOP/Passport No.", "Title of Account"]
    for field in header_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 1: Principal Applicant's Details
    principal_fields = ["Name"]
    for field in principal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 2: Redemption Details
    redemption_fields = [
        "Name of the Fund / Investment Plan", "Type of Units", "Class of Units",
        "No. of Units", "Amount", "Certificates Issued"
    ]
    
    for field in redemption_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # CDS Account Details
    cds_fields = ["Client/House/Investor Account No.", "Participant ID/IAS ID"]
    for field in cds_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Investor Type
    investor_type_fields = ["Individual Investor", "Institutional Investor"]
    for field in investor_type_fields:
        value = extract_value(response_text, field)
        if value:
            data["Investor Type"] = field
    
    return data

import re

tick_symbols = r"[✓✔☑■●▣◆■○]"


def checkbox_selected(label: str, text: str) -> bool:
    """
    Detect checkbox selections even with messy OCR.
    Matches:
        ✓ LABEL
        LABEL ✓
        [✓] LABEL
        ■ LABEL
        ● LABEL
    """
    patterns = [
        rf"{label}\s*[:\-]?\s*\(?\s*{tick_symbols}\s*\)?",
        rf"{tick_symbols}\s*\(?\s*{label}\s*\)?",
        rf"\[\s*{tick_symbols}\s*\]\s*{label}",
        rf"{label}\s*\[\s*{tick_symbols}\s*\]",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def parse_mcb_early_redemption(response_text: str) -> dict:
    """Parse MCB Early Redemption Request Form"""
    data = {}

    # ============================
    # 1) FUND SELECTION - MORE AGGRESSIVE DETECTION
    # ============================
    
    # Convert to uppercase for easier matching
    text_upper = response_text.upper()
    
    # Look for explicit mentions with checkmarks
    if "PAKISTAN PENSION FUND" in text_upper and any(marker in text_upper for marker in ["✓", "✔", "☑", "[✓]", "[X]", "SELECTED", "CHECKED"]):
        data["Fund Type"] = "PAKISTAN PENSION FUND"
    elif "ALHAMRA ISLAMIC PENSION FUND" in text_upper and any(marker in text_upper for marker in ["✓", "✔", "☑", "[✓]", "[X]", "SELECTED", "CHECKED"]):
        data["Fund Type"] = "ALHAMRA ISLAMIC PENSION FUND"
    else:
        # Fallback: check which fund is mentioned more prominently
        pakistan_count = text_upper.count("PAKISTAN PENSION FUND")
        alhamra_count = text_upper.count("ALHAMRA ISLAMIC PENSION FUND")
        
        if pakistan_count > alhamra_count:
            data["Fund Type"] = "PAKISTAN PENSION FUND (mentioned)"
        elif alhamra_count > pakistan_count:
            data["Fund Type"] = "ALHAMRA ISLAMIC PENSION FUND (mentioned)"
        else:
            data["Fund Type"] = "Not detected"

    # ============================
    # 2) HEADER FIELDS
    # ============================
    header_fields = ["Date", "Registration Number", "NTN No.", "Participant's Name"]
    for field in header_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value

    # ============================
    # 3) PARTICIPANT DETAILS
    # ============================
    participant_fields = ["Participant's Name", "Distinctive Account Number"]
    for field in participant_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value

    # ============================
    # 4) UNIT TYPE CHECKBOXES
    # ============================
    text_upper = response_text.upper()
    
    if "PROVIDENT FUND" in text_upper and any(marker in text_upper for marker in ["✓", "✔", "☑", "[✓]", "[X]", "SELECTED", "CHECKED"]):
        data["Unit Type"] = "Provident Fund"
    elif "NON PROVIDENT FUND" in text_upper and any(marker in text_upper for marker in ["✓", "✔", "☑", "[✓]", "[X]", "SELECTED", "CHECKED"]):
        data["Unit Type"] = "Non Provident Fund"
    else:
        fallback = extract_value(response_text, "Unit Type")
        data["Unit Type"] = fallback if fallback else "Not detected"

    # ============================
    # 5) REDEMPTION AMOUNT FIELDS
    # ============================
    redemption_fields = [
        "Amount to be redeemed (Rupees)",
        "In Words",
        "In Figures"
    ]

    for field in redemption_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value

    return data


def save_mcb_early_redemption_to_pdf(form_data):
    """Save MCB Early Redemption form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "mcb_early_redemption_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "MCB Early Redemption Request Form")
    
    # FUND SELECTION (TOP OF FORM)
    y = PAGE_HEIGHT - 80
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FUND SELECTION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    if "Fund Type" in form_data:
        c.drawString(40, y, f"Selected Fund: {form_data['Fund Type']}")
        y -= 15
    
    # Header Information
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "HEADER INFORMATION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    header_fields = ["Date", "Registration Number", "NTN No.", "Participant's Name"]
    for field in header_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Participant Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "PARTICIPANT DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    if "Distinctive Account Number" in form_data:
        c.drawString(40, y, f"Distinctive Account Number: {form_data['Distinctive Account Number']}")
        y -= 15
    
    # Redemption Information - UNIT TYPE FIRST
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "EARLY REDEMPTION INFORMATION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    
    # Unit Type (checkboxes come first in Section 2)
    if "Unit Type" in form_data:
        c.drawString(40, y, f"Unit Type: {form_data['Unit Type']}")
        y -= 15
    
    # Amount fields
    redemption_fields = ["Amount to be redeemed (Rupees)", "In Words", "In Figures"]
    for field in redemption_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Income Payment Plan
    if "Income Payment Plan" in form_data:
        c.drawString(40, y, f"Income Payment Plan: {form_data['Income Payment Plan']}")
        y -= 15
    
    # Tax Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TAX DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    tax_fields = [
        "Total Taxable Income for Three Preceding Tax Years", 
        "Total Tax Paid or Payable for Three Preceding Tax Years",
        "Income Tax Returns"
    ]
    for field in tax_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    c.save()
    return pdf_file

def process_mcb_early_redemption(uploaded_file, col2):
    """Process MCB Early Redemption Request Form"""
    mcb_early_redemption_prompt = """
    Extract ALL information from this MCB Early Redemption Request Form. Focus on these specific sections:

    FUND SELECTION (TOP OF FORM):
    - PAKISTAN PENSION FUND [Check if selected]
    - ALHAMRA ISLAMIC PENSION FUND [Check if selected]

    HEADER INFORMATION:
    - Date
    - Registration Number
    - NTN No.
    - Participant's Name

    SECTION 1: PARTICIPANT'S DETAILS:
    - Participant's Name
    - Distinctive Account Number

    SECTION 2: EARLY REDEMPTION INFORMATION:
    - Unit Type (Provident Fund / Non Provident Fund)
    - Amount to be redeemed (Rupees) - in Figures and in Words
    - Income Payment Plan [Check if selected]

    SECTION 3: DETAILS OF TAX (Mandatory for Non Provident Fund Unit Type):
    - Total Taxable Income for Three Preceding Tax Years
    - Total Tax Paid or Payable for Three Preceding Tax Years
    - Income Tax Returns (Attached/Not Attached)

    Extract all text exactly as it appears on the form. Include any checked boxes, selected options, or filled amounts.
    """
    
    response_key = "mcb_early_redemption_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from MCB Early Redemption Form..."):
            response = call_qwen_model_with_image(uploaded_file, mcb_early_redemption_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_mcb_early_redemption(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("MCB Early Redemption Form Data")
            with st.container(height=700):
                with st.form("mcb_early_redemption_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Fund Type", "Unit Type", "Income Payment Plan"]:
                                if field == "Fund Type":
                                    options = ["PAKISTAN PENSION FUND", "ALHAMRA ISLAMIC PENSION FUND", "Not Selected"]
                                elif field == "Unit Type":
                                    options = ["Provident Fund", "Non Provident Fund", "Income Payment Plan", "Not Selected"]
                                elif field == "Income Payment Plan":
                                    options = ["Selected", "Not Selected"]
                                
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            elif field == "Income Tax Returns":
                                options = ["Attached", "Not Attached"]
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from MCB Early Redemption Form')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_mcb_early_redemption_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="mcb_early_redemption_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("MCB Early Redemption Form data saved successfully!")

def save_mcb_redemption_c1_to_pdf(form_data):
    """Save MCB Redemption C-1 form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "mcb_redemption_c1_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "MCB Redemption Request Form C-1")
    
    # Header Information
    y = PAGE_HEIGHT - 80
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "HEADER INFORMATION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    header_fields = ["Date", "Investor Registration Number", "CNIC/NICOP/Passport No.", "Title of Account"]
    for field in header_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Principal Applicant Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "PRINCIPAL APPLICANT DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    if "Name" in form_data:
        c.drawString(40, y, f"Name: {form_data['Name']}")
        y -= 15
    
    # Redemption Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "REDEMPTION DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    redemption_fields = ["Name of the Fund / Investment Plan", "Type of Units", "Class of Units", 
                        "No. of Units", "Amount", "Certificates Issued"]
    for field in redemption_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    c.save()
    return pdf_file

def save_mcb_early_redemption_to_pdf(form_data):
    """Save MCB Early Redemption form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "mcb_early_redemption_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "MCB Early Redemption Request Form")
    
    # Fund Selection
    y = PAGE_HEIGHT - 80
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FUND SELECTION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    if "Fund Type" in form_data:
        c.drawString(40, y, f"Fund Type: {form_data['Fund Type']}")
        y -= 15
    
    # Header Information
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "HEADER INFORMATION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    header_fields = ["Date", "Registration Number", "NTN No.", "Participant's Name"]
    for field in header_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Participant Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "PARTICIPANT DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    participant_fields = ["Participant's Name", "Distinctive Account Number"]
    for field in participant_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Early Redemption Information
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "EARLY REDEMPTION INFORMATION")
    
    y -= 20
    c.setFont("Helvetica", 10)
    redemption_fields = ["Unit Type", "Amount to be redeemed (Rupees)", "In Words", "In Figures", "Income Payment Plan"]
    for field in redemption_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    # Tax Details
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TAX DETAILS")
    
    y -= 20
    c.setFont("Helvetica", 10)
    tax_fields = [
        "Total Taxable Income for Three Preceding Tax Years", 
        "Total Tax Paid or Payable for Three Preceding Tax Years",
        "Income Tax Returns"
    ]
    for field in tax_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    c.save()
    return pdf_file

# ---------- Process MCB Redemption Forms ----------
def process_mcb_redemption_c1(uploaded_file, col2):
    """Process MCB Redemption Request Form C-1"""
    mcb_redemption_prompt = """
    Extract ALL information from this MCB Redemption Request Form C-1. Focus on these specific sections:

    HEADER INFORMATION:
    - Date
    - Investor Registration Number
    - CNIC/NICOP/Passport No.
    - Title of Account

    SECTION 1: PRINCIPAL APPLICANT'S DETAILS:
    - Name (as per CNIC/NICOP/Passport)

    SECTION 2: REDEMPTION DETAILS:
    - Name of the Fund / Investment Plan
    - Type of Units
    - Class of Units
    - No. of Units
    - Amount
    - Certificates Issued (Yes/No, if yes Certificate No.)

    CDS ACCOUNT DETAILS:
    - Client/House/Investor Account No.
    - Participant ID/IAS ID

    INVESTOR TYPE:
    - Individual Investor
    - Institutional Investor

    Extract all text exactly as it appears on the form. Include any checked boxes or selected options.
    """
    
    response_key = "mcb_redemption_c1_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from MCB Redemption Form C-1..."):
            response = call_qwen_model_with_image(uploaded_file, mcb_redemption_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_mcb_redemption_c1(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("### 📝 MCB Redemption Form C-1 Data")
            with st.container(height=700):
                with st.form("mcb_redemption_c1_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Certificates Issued", "Individual Investor", "Institutional Investor"]:
                                options = ["Yes", "No"] if field == "Certificates Issued" else ["Selected", "Not Selected"]
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from MCB Redemption Form C-1')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_mcb_redemption_c1_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="mcb_redemption_c1_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("MCB Redemption Form C-1 data saved successfully!")

# ---------- Meezan Bank Specific Functions ----------
def parse_meezan_account_opening_response(response_text: str) -> dict:
    """Parse Meezan Bank Account Opening Form - Principal Account Holder"""
    data = {}
    
    # Date and Account Type
    date_fields = ["Day", "Month", "Year", "Type of Account"]
    for field in date_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Principal Account Holder Information
    principal_fields = [
        "Name", "Father's/Husband Name", "Mother's Maiden Name", 
        "CNIC/NICOP/Passport No", "Issuance Date", "Expiry Date",
        "Date of Birth", "Marital Status", "Religion", "Place of Birth",
        "Nationality", "Dual Nationality"
    ]
    
    for field in principal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Address Information
    address_fields = [
        "Mailing Address: Street", "Mailing Address: City", "Mailing Address: Country",
        "Current Address: Street", "Current Address: City", "Current Address: Country"
    ]
    
    for field in address_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def parse_meezan_account_info(response_text: str) -> dict:
    """Parse Meezan Bank Contact Details and Additional Information"""
    data = {}
    
    # Contact Details
    contact_fields = [
        "Residential Status", "Email", "Mobile Network", "Tel/Res Office", "Mobile"
    ]
    
    for field in contact_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Minor Account Information
    minor_fields = [
        "Name of Guardian", "Relation with Principal", 
        "Guardian CNIC", "CNIC Expiry Date"
    ]
    
    for field in minor_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Bank Account Details
    bank_fields = [
        "Bank Account No.", "Bank", "Branch", "City"
    ]
    
    for field in bank_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Joint Account Holders
    joint_holders = extract_meezan_joint_holders(response_text)
    if joint_holders:
        data["Joint Account Holders"] = joint_holders
    
    return data

def parse_meezan_special_instructions(response_text: str) -> dict:
    """Parse Meezan Bank Special Instructions"""
    data = {}
    
    # Special Instructions
    instruction_fields = [
        "Account Operating Instructions", "Dividend Mandate", 
        "Communication Mode", "Stock Dividend"
    ]
    
    for field in instruction_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # MTPF Account Details
    mtpf_fields = [
        "Expected Retirement Date", "Allocation Scheme"
    ]
    
    for field in mtpf_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def extract_meezan_joint_holders(response_text: str) -> dict:
    joint_data = {}

    for i in range(1, 3):
        # Extract the section for Joint Holder i
        pattern = rf"Joint Holder {i}[:\s\*]*([\s\S]*?)(?=Joint Holder {i+1}|$)"
        match = re.search(pattern, response_text, re.IGNORECASE)

        if not match:
            continue

        section = match.group(1)
        holder_data = {}

        fields = [
            "Relation with Principal",
            "Customer ID",
            "Name",
            "CNIC/NICOP/Passport",
            "Issuance Date",
            "Expiry Date"
        ]

        for field in fields:
            value = extract_value(section, field)   # <-- search inside section only
            if value is not None:
                holder_data[field] = value

        if holder_data:
            joint_data[f"Joint Holder {i}"] = holder_data

    return joint_data if joint_data else None

def save_meezan_to_pdf(data_dict1, data_dict2, data_dict3):
    """Save Meezan Bank form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "meezan_bank_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "Meezan Bank Account Opening Form - Extracted Data")
    
    # Date and Account Type
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_HEIGHT - 80, f"Date: {data_dict1.get('Day', '')}/{data_dict1.get('Month', '')}/{data_dict1.get('Year', '')}")
    c.drawString(2 * PAGE_WIDTH / 3, PAGE_HEIGHT - 80, f"Type of Account: {data_dict1.get('Type of Account', '')}")
    
    # Principal Account Holder Section
    y = PAGE_HEIGHT - 110
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "PRINCIPAL ACCOUNT HOLDER")
    
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Name: {data_dict1.get('Name', '')}")
    y -= 15
    c.drawString(40, y, f"Father/Husband Name: {data_dict1.get('Father/Husband Name', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Mother's Maiden Name: {data_dict1.get('Mother Maiden Name', '')}")
    y -= 15
    c.drawString(40, y, f"CNIC/NICOP/Passport No: {data_dict1.get('CNIC/NICOP/Passport No', '')}")
    y -= 15
    c.drawString(40, y, f"Issuance Date: {data_dict1.get('Issuance Date', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Expiry Date: {data_dict1.get('Expiry Date', '')}")
    y -= 15
    c.drawString(40, y, f"Date of Birth: {data_dict1.get('Date of Birth', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Marital Status: {data_dict1.get('Marital Status', '')}")
    y -= 15
    c.drawString(40, y, f"Religion: {data_dict1.get('Religion', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Place of Birth: {data_dict1.get('Place of Birth', '')}")
    y -= 15
    c.drawString(40, y, f"Nationality: {data_dict1.get('Nationality', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Dual Nationality: {data_dict1.get('Dual Nationality', '')}")
    
    # Address Information
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "ADDRESS INFORMATION")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, "Mailing Address:")
    y -= 15
    c.drawString(60, y, f"Street: {data_dict1.get('Mailing Address: Street', '')}")
    y -= 15
    c.drawString(60, y, f"City: {data_dict1.get('Mailing Address: City', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Country: {data_dict1.get('Mailing Address: Country', '')}")
    y -= 15
    c.drawString(40, y, "Current Address:")
    y -= 15
    c.drawString(60, y, f"Street: {data_dict1.get('Current Address: Street', '')}")
    y -= 15
    c.drawString(60, y, f"City: {data_dict1.get('Current Address: City', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Country: {data_dict1.get('Current Address: Country', '')}")
    
    # Contact Details
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "CONTACT DETAILS")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Residential Status: {data_dict2.get('Residential Status', '')}")
    y -= 15
    c.drawString(40, y, f"Email: {data_dict2.get('Email', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Mobile Network: {data_dict2.get('Mobile Network', '')}")
    y -= 15
    c.drawString(40, y, f"Mobile: {data_dict2.get('Mobile', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Tel/Res Office: {data_dict2.get('Tel/Res Office', '')}")
    
    # Special Instructions
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "SPECIAL INSTRUCTIONS")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Account Operating Instructions: {data_dict3.get('Account Operating Instructions', '')}")
    y -= 15
    c.drawString(40, y, f"Dividend Mandate: {data_dict3.get('Dividend Mandate', '')}")
    c.drawString(PAGE_WIDTH / 2, y, f"Stock Dividend: {data_dict3.get('Stock Dividend', '')}")
    y -= 15
    c.drawString(40, y, f"Communication Mode: {data_dict3.get('Communication Mode', '')}")
    
    c.save()
    return pdf_file

# ---------- MCB Bank Specific Functions ----------
def parse_mcb_funds_info(response_text: str) -> dict:
    """Parse MCB Funds Account Opening Form"""
    data = {}
    
    # Form header information
    header_fields = ["DATE", "Purpose", "Investor Registration Number"]
    for field in header_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 1: Principal Applicant's Details
    principal_fields = [
        "PRINCIPAL APPLICANT'S NAME", "FATHER/SPOUSE NAME", 
        "CNIC/NICOP/PASSPORT No./B-FORM NO.", "MOTHER MAIDEN NAME",
        "GENDER", "DATE OF BIRTH", "ZAKAT DEDUCTION", 
        "PLACE OF BIRTH", "NATIONALITY"
    ]
    
    for field in principal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 2: Guardian's Details
    guardian_fields = ["GUARDIAN NAME", "GUARDIAN CNIC/NICOP/PASSPORT No.", "RELATIONSHIP WITH MINOR"]
    for field in guardian_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 3: Contact Details
    contact_fields = [
        "RESIDENTIAL ADDRESS", "RESIDENTIAL CITY/DISTRICT", "RESIDENTIAL POSTAL CODE", "RESIDENTIAL COUNTRY",
        "OFFICE/BUSINESS ADDRESS", "OFFICE CITY/DISTRICT", "OFFICE POSTAL CODE", "OFFICE COUNTRY",
        "MAILING ADDRESS", "TELEPHONE No. RES", "TELEPHONE No. OFF", "TELEPHONE EXT",
        "FAX No.", "EMAIL ADDRESS", "MOBILE No."
    ]
    
    for field in contact_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 4: Statement Delivery
    statement_fields = ["STATEMENT DELIVERY"]
    for field in statement_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 5: Bank Details
    bank_fields = [
        "BANK ACCOUNT TITLE", "COMPLETE BANK ACCOUNT No.", 
        "BRANCH NAME & ADDRESS", "BANK NAME", "CITY", "IBAN"
    ]
    
    for field in bank_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def save_mcb_to_pdf(form_data):
    """Save MCB Funds form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "mcb_funds_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "MCB Funds - Account Opening Form")
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_HEIGHT - 70, f"Date: {form_data.get('DATE', '')}")
    c.drawString(40, PAGE_HEIGHT - 85, f"Purpose: {form_data.get('Purpose', '')}")
    c.drawString(40, PAGE_HEIGHT - 100, f"Investor Registration No: {form_data.get('Investor Registration Number', '')}")
    
    # Section 1: Principal Applicant's Details
    y = PAGE_HEIGHT - 130
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "1. PRINCIPAL APPLICANT'S DETAILS")
    
    y -= 25
    c.setFont("Helvetica", 10)
    principal_fields = [
        "PRINCIPAL APPLICANT'S NAME", "FATHER/SPOUSE NAME", 
        "CNIC/NICOP/PASSPORT No./B-FORM NO.", "MOTHER MAIDEN NAME",
        "GENDER", "DATE OF BIRTH", "ZAKAT DEDUCTION", 
        "PLACE OF BIRTH", "NATIONALITY"
    ]
    
    for field in principal_fields:
        if field in form_data:
            c.drawString(40, y, f"{field}: {form_data[field]}")
            y -= 15
    
    c.save()
    return pdf_file

# ---------- Allied Bank Specific Functions ----------
def parse_allied_personal_info(response_text: str) -> dict:
    """Parse personal information section from Allied Bank form"""
    data = {}
    
    # Personal Information fields
    personal_fields = [
        "Full Name", "Father's Name", "Mother's Maiden Name", "Date of Birth",
        "Spouse's Name", "Place of Birth", "Nationality", "Resident Status",
        "Country of Residence", "CNIC#", "Passport / Alien Card / POC #",
        "Marital Status", "Education", "Profession", "Occupation"
    ]
    
    for field in personal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Employment details
    employment_fields = [
        "Nature of Business", "Business/Employment Details", 
        "Name of Business/Organization", "Designation"
    ]
    
    for field in employment_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Contact information
    contact_fields = [
        "Business/Office Address", "Nearest Landmark", "Post Code",
        "Tel#", "Fax #", "Mob #", "E-mail"
    ]
    
    for field in contact_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def save_allied_to_pdf(personal_data):
    """Save Allied Bank form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "allied_bank_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "Allied Bank Account Opening Form - Extracted Data")
    
    # Personal Information Section
    y = PAGE_HEIGHT - 80
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "A. PERSONAL INFORMATION")
    
    y -= 25
    c.setFont("Helvetica", 10)
    
    # Personal details in two columns
    left_col = 40
    right_col = PAGE_WIDTH / 2 + 20
    
    fields_left = [
        "Full Name", "Father's Name", "Mother's Maiden Name", "Date of Birth",
        "Spouse's Name", "Place of Birth", "Nationality", "Resident Status"
    ]
    
    fields_right = [
        "Country of Residence", "CNIC#", "Passport / Alien Card / POC #",
        "Marital Status", "Education", "Profession", "Occupation"
    ]
    
    # Left column
    current_y = y
    for field in fields_left:
        if field in personal_data:
            c.drawString(left_col, current_y, f"{field}: {personal_data[field]}")
            current_y -= 15
    
    # Right column
    current_y = y
    for field in fields_right:
        if field in personal_data:
            c.drawString(right_col, current_y, f"{field}: {personal_data[field]}")
            current_y -= 15
    
    c.save()
    return pdf_file

# ---------- Alfalah Bank Specific Functions ----------
def parse_alfalah_investment_info(response_text: str) -> dict:
    """Parse Alfalah Investment Account Opening Form A-1"""
    data = {}
    
    # Form header information
    header_fields = ["Investor Registration No", "Account Type", "Sole Proprietorship"]
    for field in header_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 1: Principal Applicant's Details
    principal_fields = [
        "Name", "Father's/Husband's Name", "Mother's Maiden Name of Applicant",
        "Sole Proprietorship Name", "CNIC/NICOP/ARC/POC/Passport No",
        "Zakat Deduction", "Tax Status", "Date of Birth", "Issuance Date",
        "Expiry Date", "Place of Birth", "Religion", "Marital Status",
        "Nationality", "National Tax No", "Gender"
    ]
    
    for field in principal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Address Information
    address_fields = [
        "Current Mailing Address", "Current Mailing City", "Current Mailing Province", "Current Mailing Country",
        "Permanent Address", "Permanent City", "Permanent Province", "Permanent Country",
        "Business/Registered Address"
    ]
    
    for field in address_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Contact Information
    contact_fields = [
        "Tel No", "Office No", "Mobile No", "Alternative Mobile No",
        "WhatsApp No", "Email", "Investor's Signature"
    ]
    
    for field in contact_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def save_alfalah_to_pdf(form_data):
    """Save Alfalah Bank Investment form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "alfalah_investment_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "Alfalah Investments - Account Opening Form A-1")
    
    # Account Type Section
    y = PAGE_HEIGHT - 80
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "ACCOUNT TYPE AND SOLE PROPRIETORSHIP")
    
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Account Type: {form_data.get('Account Type', '')}")
    y -= 15
    c.drawString(40, y, f"Sole Proprietorship: {form_data.get('Sole Proprietorship', '')}")
    
    c.save()
    return pdf_file

# ---------- Askari Bank Specific Functions ----------
def parse_askari_investment_info(response_text: str) -> dict:
    """Parse Askari Investment Management Account Opening Form"""
    data = {}
    
    # Date Information
    date_fields = ["DATE", "Day", "Month", "Year"]
    for field in date_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 1: Principal Account Holder Information
    principal_fields = [
        "Full Name", "Father's/Husband's Name", "Mailing Address", 
        "TYPE OF INSTITUTION", "CNIC", "E-mail", "Mobile No.", 
        "Nationality", "Marital Status", "Country", "Gender", 
        "Zakat Deduction", "Date of Birth"
    ]
    
    for field in principal_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 4: Operating Instructions
    operating_fields = ["Operating Instructions", "Solely"]
    for field in operating_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 5: Other Instructions
    other_instructions_fields = [
        "Payment to be sent on registered address", 
        "Dividend Distribution option"
    ]
    
    for field in other_instructions_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    # Section 6: Declaration
    declaration_fields = ["Principal Signatory", "Authorised Signatory", "Name"]
    for field in declaration_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
    
    return data

def save_askari_to_pdf(form_data):
    """Save Askari Bank Investment form data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "askari_investment_form_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "Askari Investment Management - Account Opening Form")
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_HEIGHT - 70, f"Date: {form_data.get('DATE', '')}")
    
    # Section 1: Principal Account Holder
    y = PAGE_HEIGHT - 100
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "1. INFORMATION ABOUT THE PRINCIPAL ACCOUNT HOLDER")
    
    y -= 25
    c.setFont("Helvetica", 10)
    
    # Personal details in two columns
    left_col = 40
    right_col = PAGE_WIDTH / 2 + 20
    
    principal_fields_left = [
        "Full Name", "Father's/Husband's Name", "Mailing Address", 
        "TYPE OF INSTITUTION", "CNIC", "E-mail"
    ]
    
    principal_fields_right = [
        "Mobile No.", "Nationality", "Marital Status", "Country", 
        "Gender", "Zakat Deduction", "Date of Birth"
    ]
    
    # Left column
    current_y = y
    for field in principal_fields_left:
        if field in form_data:
            c.drawString(left_col, current_y, f"{field}: {form_data[field]}")
            current_y -= 15
    
    # Right column
    current_y = y
    for field in principal_fields_right:
        if field in form_data:
            c.drawString(right_col, current_y, f"{field}: {form_data[field]}")
            current_y -= 15
    
    c.save()
    return pdf_file

# ---------- Bank Processing Functions ----------
def process_meezan_form(uploaded_file, col2):
    """Process Meezan Bank form"""
    segments = segment_image(Image.open(uploaded_file))   # returns 3 image paths
    seg1, seg2, seg3 = segments[0], segments[1], segments[2]
    meezan_prompt1 = """Extract the following details from the form in a structured and complete manner and keep field names exact as follows: * Date: [Your answer here] * Day: [Your answer here] * Month: [Your answer here] * Year: [Your answer here] * Type of Account: [Your answer here] * Principal Account Holder: * Name: [Your answer here] * Father's/Husband Name: [Your answer here] * Mother's Maiden Name: [Your answer here] * CNIC/NICOP/Passport No: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here] * Date of Birth: [Your answer here] * Marital Status: [Single/Married] * Religion: [Muslim/Non-Muslim] * Place of Birth: [Your answer here] * Nationality: [Your answer here] * Dual Nationality: [Yes/No] * Mailing Address: * Street: [Your answer here] * City: [Your answer here] * Country: [Your answer here] * Current Address: * Street: [Your answer here] * City: [Your answer here] * Country: [Your answer here] Leave any field blank if the information is missing or not available."""
    
    meezan_prompt2 = "Extract the following details from the form in a structured and complete manner: * Residential Status: [Your answer here] * Email: [Your answer here] * Mobile Network: [Your answer here] * Tel/Res Office: [Your answer here] * Mobile: [Your answer here] * In Case of Minor Account: * Name of Guardian: [Your answer here] * Relation with Principal: [Your answer here] * Guardian CNIC: [Your answer here] * CNIC Expiry Date: [Your answer here] * Bank Account Detail: * Bank Account No.: [Your answer here] * Bank: [Your answer here] * Branch: [Your answer here] * City: [Your answer here] * Joint Account Holders: * Joint Holder 1: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here] * Joint Holder 2: * Name: [Your answer here] * Relation with Principal: [Your answer here] * Customer ID: [Your answer here] * CNIC/NICOP/Passport: [Your answer here] * Issuance Date: [Your answer here] * Expiry Date: [Your answer here]Leave blank if missing"
    
    meezan_prompt3 = """Extract the following details from the form should in a structured and complete manner:

        Special Instructions:
        Account Operating Instructions: [Principal Account Holder Only/Either or Survivor/Any Two/All]

        Dividend Mandate: [Cash / Reinvest]
        Communication Mode: [e.g., Email, Postal Mail, etc.]

        Stock Dividend: [Issue Bonus Units / Encash Bonus Units]

        Detail About Meezan Tahaffuz Pension Fund (MTPF) Account:
        Expected Retirement Date: [DD/MM/YYYY]

        Allocation Scheme: [List all ticked or checked options, e.g., Equity, Debt, Money Market. Leave blank if none are selected.]

        Please ensure only the checked/ticked values are included in the "Allocation Scheme" list. If no boxes are selected, return an empty list."""
    
    # Process Meezan form with three prompts
    if "meezan_response1" not in st.session_state:
        with st.spinner("Extracting principal account holder details..."):
            #response1 = call_openai_api_with_image(open(seg1, "rb"), meezan_prompt1)
            response1 = call_openai_api_with_image(uploaded_file, meezan_prompt1)
            #response1 = call_qwen_model_with_image(uploaded_file, meezan_prompt1)
            st.session_state.meezan_response1 = response1
    
    if "meezan_response2" not in st.session_state:
        with st.spinner("Extracting contact and joint holder details..."):
            response2 = call_openai_api_with_image(open(seg2, "rb"), meezan_prompt2)
            #response2 = call_openai_api_with_image(uploaded_file, meezan_prompt2)
            #response2 = call_qwen_model_with_image(uploaded_file, meezan_prompt2)
            st.session_state.meezan_response2 = response2
    
    if "meezan_response3" not in st.session_state:
        with st.spinner("Extracting special instructions..."):
            #response3 = call_openai_api_with_image(open(seg3, "rb"), meezan_prompt3)
            #response3 = call_openai_api_with_image(uploaded_file, meezan_prompt3)
            #response3 = call_qwen_model_with_image(open(seg3, "rb"), meezan_prompt3)
            #response3 = call_openai_api_with_image(open(seg3, "rb"), meezan_prompt3)
            response3 = call_openai_api_with_image(uploaded_file, meezan_prompt3)
            #response3 = call_qwen_model_with_image(uploaded_file, meezan_prompt3)
            # response3 = call_openai_api_with_image(open(seg3, "rb"), meezan_prompt3)
            #response3 = call_qwen_model_with_image(uploaded_file, meezan_prompt3)
            #response3 = call_qwen_model_with_image(open(seg3, "rb"), meezan_prompt3)
            print(response3)
            st.write(response3)
            st.session_state.meezan_response3 = response3
    
    if all(key in st.session_state for key in ["meezan_response1", "meezan_response2", "meezan_response3"]):
        # Parse all three responses
        parsed_data1 = parse_meezan_account_opening_response(st.session_state.meezan_response1)
        parsed_data2 = parse_meezan_account_info(st.session_state.meezan_response2)
        parsed_data3 = parse_meezan_special_instructions(st.session_state.meezan_response3)
        
        flat_data1 = flatten_dict(parsed_data1)
        flat_data2 = flatten_dict(parsed_data2)
        flat_data3 = flatten_dict(parsed_data3)
        
        with col2:
            st.markdown("### 📝 Meezan Bank Form Data")
            with st.container(height=700):
                with st.form("meezan_bank_form"):
                    edited_form1 = {}
                    edited_form2 = {}
                    edited_form3 = {}
                    
                    cols = st.columns(2)
                    
                    # Section 1: Principal Account Holder
                    st.subheader("Principal Account Holder")
                    for idx, (field, value) in enumerate(flat_data1.items()):
                        col = cols[idx % 2]
                        with col:
                            edited_form1[field] = st.text_input(field, value=value if value else "")
                    
                    # Section 2: Contact Details
                    st.subheader("Contact Details")
                    for idx, (field, value) in enumerate(flat_data2.items()):
                        col = cols[idx % 2]
                        with col:
                            edited_form2[field] = st.text_input(field, value=value if value else "")
                    
                    # Section 3: Special Instructions
                    st.subheader("Special Instructions")
                    for idx, (field, value) in enumerate(flat_data3.items()):
                        col = cols[idx % 2]
                        with col:
                            edited_form3[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        # Display extracted data
        st.subheader('Principal Account Holder')
        st.write(st.session_state.meezan_response1)
        
        st.subheader('Contact Details')
        display_dict(parsed_data2)
        
        st.subheader('Special Instructions')
        display_dict(parsed_data3)
        
        if submitted:
            pdf_path = save_meezan_to_pdf(edited_form1, edited_form2, edited_form3)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="meezan_bank_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("Meezan Bank Form data saved successfully!")

def process_mcb_form(uploaded_file, col2):
    """Process MCB Bank form"""
    mcb_prompt = """
    Extract ALL information from this MCB Funds Account Opening Form. Focus on these specific sections:

    FORM HEADER:
    - DATE
    - Purpose (NEW ACCOUNT OPENING, EXISTING ACCOUNT REG NO. REGULARIZATION, EXISTING ACCOUNT REG NO. UPGRADATION)
    - Investor Registration Number (for official use only)

    SECTION 1: PRINCIPAL APPLICANT'S DETAILS:
    - PRINCIPAL APPLICANT'S NAME (as per CNIC/NICOP/Passport No./B-Form No.)
    - FATHER/SPOUSE NAME (as per identity document)
    - CNIC/NICOP/PASSPORT No./B-FORM NO.
    - MOTHER MAIDEN NAME
    - GENDER (MALE, FEMALE, TRANSGENDER)
    - DATE OF BIRTH
    - ZAKAT DEDUCTION (Yes/No)
    - PLACE OF BIRTH
    - NATIONALITY

    SECTION 2: GUARDIAN'S DETAILS (For Minor Applicant):
    - GUARDIAN NAME (as per CNIC/NICOP/PASSPORT No.)
    - GUARDIAN CNIC/NICOP/PASSPORT No.
    - RELATIONSHIP WITH MINOR

    SECTION 3: CONTACT DETAILS:
    - RESIDENTIAL ADDRESS (City/District, Postal Code, Country)
    - OFFICE/BUSINESS ADDRESS (City/District, Postal Code, Country)
    - MAILING ADDRESS (RESIDENTIAL ADDRESS or OFFICE/BUSINESS ADDRESS)
    - TELEPHONE No. (RES, OFF, EXT)
    - FAX No.
    - EMAIL ADDRESS
    - MOBILE No.

    SECTION 4: STATEMENT OF ACCOUNT DELIVERY:
    - Delivery Method (By Email, By Post)

    SECTION 5: BANK DETAILS:
    - BANK ACCOUNT TITLE
    - COMPLETE BANK ACCOUNT No.
    - BRANCH NAME & ADDRESS
    - BANK NAME
    - CITY
    - IBAN

    Extract all text exactly as it appears on the form.
    """
    
    response_key = "mcb_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from MCB Funds form..."):
            response = call_openai_api_with_image(uploaded_file, mcb_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_mcb_funds_info(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("### 📝 MCB Funds Form Data")
            with st.container(height=700):
                with st.form("mcb_bank_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Purpose", "GENDER", "ZAKAT DEDUCTION", "MAILING ADDRESS", "STATEMENT DELIVERY"]:
                                if field == "Purpose":
                                    options = ["NEW ACCOUNT OPENING", "EXISTING ACCOUNT REG NO. REGULARIZATION", "EXISTING ACCOUNT REG NO. UPGRADATION"]
                                elif field == "GENDER":
                                    options = ["Male", "Female", "Transgender"]
                                elif field == "ZAKAT DEDUCTION":
                                    options = ["Yes", "No"]
                                elif field == "MAILING ADDRESS":
                                    options = ["RESIDENTIAL ADDRESS", "OFFICE/BUSINESS ADDRESS"]
                                elif field == "STATEMENT DELIVERY":
                                    options = ["By Email", "By Post"]
                                else:
                                    options = [value] if value else [""]
                                
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from MCB Funds Form')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_mcb_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="mcb_funds_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("MCB Form data saved successfully!")

def process_allied_form(uploaded_file, col2):
    """Process Allied Bank form"""
    allied_prompt = """
    Extract ALL information from this Allied Bank Account Opening Form. Focus on these specific sections:

    A. PERSONAL INFORMATION:
    - Full Name (as per ID Document)
    - Father's Name
    - Mother's Maiden Name
    - Date of Birth
    - Spouse's Name
    - Place of Birth
    - Nationality
    - Resident Status (Resident/Non-Resident)
    - Country of Residence
    - CNIC# (for Pakistani Nationals)
    - Passport/Alien Card/POC # (for Foreign Nationals)
    - Marital Status (Single/Married)
    - Education (Up to Matric/O-Level, Intermediate/A-Level, Bachelors)
    - Profession (Student, Housewife, Agriculturist, Other)
    - Occupation (Self-employed, Salaried, Other)

    EMPLOYMENT DETAILS (if applicable):
    - Nature of Business (if self-employed): Retail, Trading, Services, Manufacturing, Self-employed professional, Other
    - Business/Employment Details: Federal Government, Provincial Government, Semi-Government, Private Service, Allied Bank Employee, Other
    - Name of Business/Organization
    - Designation

    CONTACT INFORMATION:
    - Business/Office Address
    - Nearest Landmark
    - Post Code
    - Tel#
    - Fax #
    - Mob #
    - E-mail

    Extract all text exactly as it appears on the form.
    """
    
    response_key = "allied_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from Allied Bank form..."):
            response = call_openai_api_with_image(uploaded_file, allied_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_allied_personal_info(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("### 📝 Allied Bank Form Data")
            with st.container(height=700):
                with st.form("allied_bank_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Marital Status", "Resident Status", "Education", "Profession", "Occupation", "Nature of Business", "Business/Employment Details"]:
                                if field == "Marital Status":
                                    options = ["Single", "Married"]
                                elif field == "Resident Status":
                                    options = ["Resident", "Non-Resident"]
                                elif field == "Education":
                                    options = ["Up to Matric/O-Level", "Intermediate/A-Level", "Bachelors"]
                                elif field == "Profession":
                                    options = ["Student", "Housewife", "Agriculturist", "Other"]
                                elif field == "Occupation":
                                    options = ["Self-employed", "Salaried", "Other"]
                                elif field == "Nature of Business":
                                    options = ["Retail", "Trading", "Services", "Manufacturing", "Self-employed professional", "Other"]
                                elif field == "Business/Employment Details":
                                    options = ["Federal Government Employee", "Provincial Government Employee", "Semi-Government Employee", "Private Service", "Allied Bank Employee", "Other"]
                                else:
                                    options = [value] if value else [""]
                                
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from Allied Bank Form')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_allied_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="allied_bank_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("Allied Bank Form data saved successfully!")

def process_alfalah_form(uploaded_file, col2):
    """Process Alfalah Bank form"""
    alfalah_prompt = """
    Extract ALL information from this Alfalah Investments Account Opening Form A-1 for Individual Investor. Focus on these specific sections:

    FORM HEADER:
    - Investor Registration No (For Office Use Only)
    - Account Type (Single, Joint, Minor, Sole Proprietorship)
    - Sole Proprietorship (Partnership, Registered, Unregistered)

    SECTION 1: PRINCIPAL APPLICANT'S DETAILS:
    - Name (Mr./Ms./Mrs.)
    - Father's/Husband's Name
    - Mother's Maiden Name of Applicant
    - Sole Proprietorship Name (if applicable)
    - CNIC/NICOP/ARC/POC/Passport No
    - Zakat Deduction (Yes/No)
    - Tax Status (Filer/Non-Filer)
    - Date of Birth
    - Issuance Date
    - Expiry Date
    - Place of Birth
    - Religion
    - Marital Status
    - Nationality
    - National Tax No (NTN)
    - Gender (Male/Female)

    ADDRESS INFORMATION:
    - Current Mailing Address (City, Province, Country)
    - Permanent Address (City, Province, Country)
    - Business/Registered Address (for sole proprietors)

    CONTACT INFORMATION:
    - Tel No (Res)
    - Office No
    - Mobile No
    - Alternative Mobile No
    - WhatsApp No
    - Email
    - Investor's Signature

    Extract all text exactly as it appears on the form.
    """
    
    response_key = "alfalah_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from Alfalah Bank form..."):
            response = call_openai_api_with_image(uploaded_file, alfalah_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_alfalah_investment_info(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("### 📝 Alfalah Bank Form Data")
            with st.container(height=700):
                with st.form("alfalah_bank_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Account Type", "Zakat Deduction", "Tax Status", "Gender", "Marital Status"]:
                                if field == "Account Type":
                                    options = ["Single", "Joint", "Minor", "Sole Proprietorship"]
                                elif field == "Zakat Deduction":
                                    options = ["Yes", "No"]
                                elif field == "Tax Status":
                                    options = ["Filer", "Non-Filer"]
                                elif field == "Gender":
                                    options = ["Male", "Female"]
                                elif field == "Marital Status":
                                    options = ["Single", "Married"]
                                else:
                                    options = [value] if value else [""]
                                
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from Alfalah Investment Form')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_alfalah_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="alfalah_investment_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("Alfalah Bank Form data saved successfully!")

def process_askari_form(uploaded_file, col2):
    """Process Askari Bank form"""
    askari_prompt = """
    You are an OCR and form-structure extraction system.
Your task is to extract information ONLY from what is clearly visible on the provided image of an
“Askari Investment Management – Account Opening Form”.

CRITICAL RULES (MUST FOLLOW):
1. Do NOT guess, infer, assume, or auto-complete any information.
2. Do NOT hallucinate checkbox selections.
3. A checkbox value should be marked ONLY if a visible tick (✔ / ✓) or clear mark is present.
4. If a field is blank, unclear, overwritten, or not visible, write exactly: "Not provided".
5. Extract text EXACTLY as written (including spelling, spacing, and capitalization).
6. Do NOT add explanations, summaries, or interpretations.

-------------------------
OUTPUT FORMAT (STRICT)
-------------------------

Return the output as a VALID JSON object using the exact structure and keys below.
Do NOT add extra keys.
Do NOT remove keys.
Do NOT change key names.

{
  "document_info": {
    "document_title": "",
    "date": {
      "day": "",
      "month": "",
      "year": ""
    }
  },

  "section_1_principal_account_holder": {
    "full_name": "",
    "father_or_husband_name": "",
    "mailing_address": "",
    "type_of_institution": {
      "individual": "",
      "company": "",
      "partnership": "",
      "ngo": "",
      "trust": "",
      "others": ""
    },
    "cnic": "",
    "email": "",
    "mobile_no": "",
    "nationality": "",
    "marital_status": {
      "single": "",
      "married": ""
    },
    "country": "",
    "gender": {
      "male": "",
      "female": ""
    },
    "zakat_deduction": {
      "yes": "",
      "no": ""
    },
    "date_of_birth": ""
  },

  "section_1_bank_account_details": {
    "title_of_account": "",
    "bank_account_number": "",
    "bank_name": "",
    "branch": "",
    "bank_address": "",
    "bank_telephone_no": ""
  },

  "section_2_joint_account_holders": [
    {
      "name": "",
      "cnic_or_passport_no": ""
    }
  ],

  "section_3_nominee_information": [
    {
      "name": "",
      "relationship_with_principal": "",
      "cnic_no": "",
      "percentage": ""
    }
  ],

  "section_4_operating_instructions": {
    "solely": "",
    "principal_account_holder": "",
    "jointly_any_two": "",
    "jointly_all": "",
    "either_or_survivor": "",
    "others": ""
  },

  "section_5_other_instructions": {
    "payment_sent_to_registered_address": {
      "yes": "",
      "no": ""
    },
    "dividend_distribution_option": {
      "provide_cash": "",
      "reinvest_dividend": ""
    },
    "special_instructions": ""
  },

  "section_6_declaration": {
    "principal_signatory_name": "",
    "principal_signatory_signature_present": "",
    "authorized_signatory_name": "",
    "authorized_signatory_signature_present": ""
  }
}

-------------------------
CHECKBOX OUTPUT RULE
-------------------------
For checkboxes:
- If checked → write "Checked"
- If unchecked or empty → write "Not provided"

-------------------------
FINAL REMINDER
-------------------------
If something is not clearly visible on the form image, write "Not provided".
Return ONLY the JSON output and nothing else.

    """
    
    response_key = "askari_response"
    if response_key not in st.session_state:
        with st.spinner("Extracting data from Askari Bank form..."):
            response = call_openai_api_with_image(uploaded_file, askari_prompt)
            st.session_state[response_key] = response
    
    if response_key in st.session_state:
        parsed_data = parse_askari_investment_info(st.session_state[response_key])
        flat_data = flatten_dict(parsed_data)
        
        with col2:
            st.markdown("### 📝 Askari Bank Form Data")
            with st.container(height=700):
                with st.form("askari_bank_form"):
                    edited_form = {}
                    cols = st.columns(2)
                    
                    for idx, (field, value) in enumerate(flat_data.items()):
                        col = cols[idx % 2]
                        with col:
                            if field in ["Marital Status", "Gender", "Zakat Deduction", "TYPE OF INSTITUTION", "Dividend Distribution option"]:
                                if field == "Marital Status":
                                    options = ["Single", "Married"]
                                elif field == "Gender":
                                    options = ["Male", "Female"]
                                elif field == "Zakat Deduction":
                                    options = ["Yes", "No"]
                                elif field == "TYPE OF INSTITUTION":
                                    options = ["Individual", "Corporate", "Institution"]
                                elif field == "Dividend Distribution option":
                                    options = ["Reinvest Dividend", "Credit to Bank Account", "Pay by Cheque"]
                                else:
                                    options = [value] if value else [""]
                                
                                default_index = options.index(value) if value in options else 0
                                edited_form[field] = st.selectbox(field, options, index=default_index)
                            else:
                                edited_form[field] = st.text_input(field, value=value if value else "")
                    
                    submitted = st.form_submit_button("Save Form")
        
        st.subheader('Extracted Data from Askari Investment Form')
        st.write(st.session_state[response_key])
        
        st.subheader('Structured Data')
        display_dict(parsed_data)
        
        if submitted:
            pdf_path = save_askari_to_pdf(edited_form)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    "Download PDF", 
                    f, 
                    file_name="askari_investment_form_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )
                st.success("Askari Bank Form data saved successfully!")

# ---------- Al Meezan Package Specific Functions ----------
def call_openai_api_with_image_single(image, prompt, page_num):
    """Call OpenAI API with a single image"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error calling OpenAI API for page {page_num}: {str(e)}")
        return None
    
def call_qwen_api_with_image_single(image, prompt, page_num,
                                    model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic"):
    """Call Qwen Vision model with a single PIL image"""
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=st.secrets["HF_TOKEN"],
        )

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error calling Qwen API for page {page_num}: {str(e)}")
        return None

# Page number (1-based) → form key
AL_MEEZAN_PAGE_GROUPS = {
    "account_opening": [1, 2, 3],
    "fatca": [4],
    "crs": [5],
}


def merge_images_vertically(images):
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)

    merged = Image.new("RGB", (max_width, total_height), color="white")

    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height

    return merged


def get_account_opening_prompt_by_page(page_num, base_prompt):
    if page_num == 1:
        return """
        Extract ONLY the following sections from the INVESTOR ACCOUNT OPENING FORM FOR INDIVIDUAL (Page 1).

        Extract ALL fields exactly as written.
        For checkboxes, return ONLY the selected option.
        If nothing is selected, write: Not Selected
        Format strictly as:
        Field Name: Value


        ========================
        SECTION 1: ACCOUNT INFORMATION
        ========================
        - Customer ID:
        - Portfolio No:
        - Day:
        - Month:
        - Year:
        - Type of Account: [Single / Joint / Minor / MTPF]


        ========================
        SECTION 2: PRINCIPAL ACCOUNT HOLDER
        ========================
        - Name:
        - Title (Mr./Mrs./Ms.):
        - Father's/Husband's Name:
        - Mother's Maiden Name:
        - CNIC/NICOP/Passport No:
        - Issuance Date:
        - Expiry Date:
        - Marital Status: [Single / Married]
        - Religion: [Muslim / Non Muslim]
        - Place of Birth:
        - Date of Birth:
        - Nationality:
        - Dual Nationality: [Yes / No]
        - If Yes, specify:


        ========================
        SECTION 3: ADDRESSES
        ========================
        - Mailing Address:
        - Mailing City:
        - Mailing Country:
        - Current Address (as per CNIC):
        - Current City:
        - Current Country:


        ========================
        SECTION 4: RESIDENTIAL STATUS
        ========================
        - Residential Status:
        [Pakistan Resident / Non-Resident / Resident Foreign National / Non-Resident Foreign National]


        ========================
        SECTION 5: CONTACT DETAILS
        ========================
        - Email:
        - Tel Res/Office:
        - Mobile:
        - Mobile Network:


        ========================
        SECTION 6: IN CASE OF MINOR ACCOUNT
        ========================
        - Name of Guardian:
        - Relation with Principal:
        - Guardian CNIC:
        - Guardian CNIC Expiry Date:


        ========================
        SECTION 7: BANK ACCOUNT DETAIL OF PRINCIPAL ACCOUNT HOLDER
        (For Redemption and Dividend Payments)
        ========================
        - Bank Account No (IBAN preferred):
        - Bank Name:
        - Branch:
        - City:


        ========================
        SECTION 8: JOINT ACCOUNT HOLDERS
        ========================

        ---- Joint Holder 1 ----
        - Joint Holder 1 Name:
        - Relation with Principal:
        - Customer ID (if any):
        - CNIC/NICOP/Passport:
        - Issuance Date:
        - Expiry Date:

        ---- Joint Holder 2 ----
        - Joint Holder 2 Name:
        - Relation with Principal:
        - Customer ID (if any):
        - CNIC/NICOP/Passport:
        - Issuance Date:
        - Expiry Date:


        ========================
        SECTION 9: SPECIAL INSTRUCTIONS
        ========================
        - Account Operating Instructions:
        [Principal Account Holder Only / Either or Survivor / Any Two / All]

        - Dividend Mandate:
        [Cash / Reinvest]

        - Stock Dividend:
        [Issue Bonus Units / Encash Bonus Units]

        - Communication Mode:
        [Electronic Only / Physical Communication]


        ========================
        SECTION 10: DETAIL ABOUT MEEZAN TAHAFFUZ PENSION FUND (MTPF)
        ========================
        - Expected Retirement Date (DD/MM/YYYY):
        - Note for Pension Fund investments over Rs. 3 million:


        ========================
        SECTION 11: ALLOCATION SCHEME SELECTION
        (Extract ONLY the checked option)
        ========================
        - Selected Allocation Scheme:
        [High Volatility /
        High Volatility with Gold /
        Medium Volatility /
        Medium Volatility with Gold /
        Low Volatility /
        Low Volatility with Gold /
        Lower Volatility /
        Lower Volatility with Gold /
        Life Cycle Plan /
        100% Debt /
        100% Equity /
        100% Money Market /
        100% Gold]
        """

    elif page_num == 2:
        return """
        Extract ONLY the following sections from the INVESTOR ACCOUNT OPENING FORM:

        SECTION 9: SOURCE OF INCOME & WEALTH
            - Source of Income: 
            - Source of Wealth: 
            - Name of Employer/Business (if Applicable): 
            - Designation: 
            - Nature of Business: 
            
        SECTION 10: EDUCATION & GEOGRAPHY
        - Education: 
        - Geographies involved: 
        - Type of Counterparties: 
            
        SECTION 11: TRANSACTION DETAILS
        - Possible Modes of Transactions: 
        - Expected Turnover in Account: 
        - Expected Amount of Investment: 
        - Annual Income: 
        - Expected No. of Transactions: 
        
        SECTION 12: RISK ASSESSMENT
        - Age Group: 
        - Risk-Return Tolerance: 
        - Monthly Savings: 
        - Occupation: 
        
        SECTION 13: INVESTMENT KNOWLEDGE
        - Investment Knowledge Level: 
        - Investment Objective: 
        - Investment Horizon: 
        
        SECTION 14: INVESTOR PORTFOLIO CALCULATION
        - Total Score: 
        - Recommended Portfolio: 
        - Calculated ideal Portfolio: 
    
        SECTION 15: NEXT OF KIN
        - Next of Kin Name: 
        - Next of Kin Contact: 
        - Next of Kin Address: 
        
        SECTION 16: BENEFICIARY DETAILS
        - Ultimate Beneficiary Name: 
        - Relation with Customer: 
        - Beneficiary CNIC/NICOP/Passport No: 

        IMPORTANT: Extract ALL fields exactly as they appear. For checkboxes, write the SELECTED option only.
        Format: Field Name: Value
        """

    elif page_num == 3:
        return """
        Extract ONLY the following sections from the INVESTOR ACCOUNT OPENING FORM:

        SECTION 17: GUIDELINES FOR INVESTORS
            - Guidelines Read and Understood: [Yes/No]
            
        SECTION 18: NOTE AND DECLARATION
        - Declaration Signed: [Yes/No]
        
        SECTION 19: APPLICATION CHECKLIST
        - CNIC Copy Attached: [Yes/No]
        - Business/Employment Proof Attached: [Yes/No]
        - Zakat Declaration Attached: [Yes/No]
        - FATCA Form Attached: [Yes/No]
        - CRS Form Attached: [Yes/No]
        - Health Questionnaire Attached: [Yes/No]
        
        SECTION 20: SALES INFORMATION
        - Sales Person's Name: 
        - Manager's Name: 
        - Distributor Stamp: [Present/Absent]
        - DAO Code: 
        - Remarks: 

        IMPORTANT: Extract ALL fields exactly as they appear. For checkboxes, write the SELECTED option only.
        Format: Field Name: Value
        """

    return base_prompt

# ---------- Al Meezan Package Specific Functions ----------
def process_al_meezan_package(uploaded_file, col2):
    """Process Al Meezan Full Package with detailed extraction"""
    st.markdown("## 🏦 Al Meezan - Forms Package")
    st.info("Extracting 3 Forms from Package")
    try:
        current_hash = file_hash(uploaded_file)
    except Exception:
        current_hash = None

    # If we've already extracted for this uploaded file (and hash matches), reuse cached data
    if current_hash and st.session_state.get("al_meezan_file_hash") == current_hash and "al_meezan_extracted_data" in st.session_state:
        extracted_data = st.session_state["al_meezan_extracted_data"]
        # ensure PDF path exists (generate if missing)
        if "al_meezan_pdf_path" not in st.session_state:
            try:
                # combine the three forms into one dict for PDF generation
                combined = {}
                combined.update(extracted_data.get("account_opening", {}))
                combined.update(extracted_data.get("fatca", {}))
                combined.update(extracted_data.get("crs", {}))
                pdf_path = save_al_meezan_package_to_pdf(combined)
                st.session_state["al_meezan_pdf_path"] = pdf_path
            except Exception:
                pass

        display_al_meezan_package_tabbed(extracted_data, None, col2)
        return
    # Convert PDF to images
    with st.spinner("📄 Converting PDF to images..."):
        pdf_images = convert_pdf_to_images(uploaded_file)
        if not pdf_images:
            st.error("❌ Failed to convert PDF")
            return
        
        st.success(f"✅ Converted {len(pdf_images)} pages")
    
    # Define form-specific prompts (detailed prompts for each form)
    form_prompts = {
        "account_opening": {
            "name": "Investor Account Opening Form for Individual",
            "keywords": ["account opening", "customer id", "portfolio no", "cnic", "passport", "date of birth", "address"],
            "prompt": """
            Extract ALL information from the INVESTOR ACCOUNT OPENING FORM FOR INDIVIDUAL.
            
            SECTION 1: ACCOUNT INFORMATION
            - Customer ID: 
            - Portfolio No: 
            - Date: [DD/MM/YYYY]
            - Day: [DD]
            - Month: [MM]
            - Year: [YYYY]
            - Type of Account: 
            
            SECTION 2: PERSONAL DETAILS
            - Name: 
            - Title (Mr./Mrs./Ms.): 
            - Father's/Husband's Name: 
            - Mother's Maiden Name: 
            - CNIC/NICOP/Passport No: 
            - Issuance Date: [DD/MM/YYYY]
            - Expiry Date: [DD/MM/YYYY]
            - Date of Birth: [DD/MM/YYYY]
            - Marital Status: 
            - Religion: 
            - Place of Birth: 
            - Nationality: 
            - Dual Nationality: 
            
            SECTION 3: ADDRESSES
            - Mailing Address: 
            - Mailing City: 
            - Mailing Country: 
            - Current Address: 
            - Current City: 
            - Current Country: 
            
            SECTION 4: CONTACT DETAILS
            - Residential Status: 
            - Email: 
            - Mobile: 
            - Mobile Network: 
            - Tel/Res Office: 
            
            SECTION 5: BANK DETAILS
            - Bank Account No: 
            - Bank Name: 
            - Branch: 
            - City: 
            
            SECTION 6: IN CASE OF MINOR ACCOUNT
            - Name of Guardian:
            - Relation with Principal:
            - Guardian CNIC:
            - Guardian CNIC Expiry Date:

            SECTION 7: JOINT ACCOUNT HOLDERS
            ---- Joint Holder 1 ----
            - Joint Holder 1 Name:
            - Relation with Principal:
            - Customer ID (if any):
            - CNIC/NICOP/Passport:
            - Issuance Date:
            - Expiry Date:

            ---- Joint Holder 2 ----
            - Joint Holder 2 Name:
            - Relation with Principal:
            - Customer ID (if any):
            - CNIC/NICOP/Passport:
            - Issuance Date:
            - Expiry Date:

            SECTION 8: SPECIAL INSTRUCTIONS
            - Account Operating Instructions: 
            - Dividend Mandate: 
            - Communication Mode: 
            - Stock Dividend: 

            SECTION 9: DETAIL ABOUT MEEZAN TAHAFFUZ PENSION FUND (MTPF) ACCOUNT
            - Expected Retirement Date (DD/MM/YYYY): 
            - Note for Pension Fund investments over Rs. 3 million: 
            
            SECTION 10: ALLOCATION SCHEME SELECTION
            - Selected Allocation Scheme: [Extract the checked option only]
            
            SECTION 11: SOURCE OF INCOME & WEALTH
            - Source of Income: 
            - Source of Wealth: 
            - Name of Employer/Business (if Applicable): 
            - Designation: 
            - Nature of Business: 
            
            SECTION 12: EDUCATION & GEOGRAPHY
            - Education: 
            - Geographies involved: 
            - Type of Counterparties: 
            
            SECTION 13: TRANSACTION DETAILS
            - Possible Modes of Transactions: 
            - Expected Turnover in Account: 
            - Expected Amount of Investment: 
            - Annual Income: 
            - Expected No. of Transactions: 
            
            SECTION 14: RISK ASSESSMENT
            - Age Group: 
            - Risk-Return Tolerance: 
            - Monthly Savings: 
            - Occupation: 
            
            SECTION 15: INVESTMENT KNOWLEDGE
            - Investment Knowledge Level: 
            - Investment Objective: 
            - Investment Horizon: 
            
            SECTION 16: INVESTOR PORTFOLIO CALCULATION
            - Total Score: 
            - Recommended Portfolio: 
            - Calculated ideal Portfolio: 
        
            SECTION 17: NEXT OF KIN
            - Next of Kin Name: 
            - Next of Kin Contact: 
            - Next of Kin Address: 
            
            SECTION 18: BENEFICIARY DETAILS
            - Ultimate Beneficiary Name: 
            - Relation with Customer: 
            - Beneficiary CNIC/NICOP/Passport No: 
       
            SECTION 19: GUIDELINES FOR INVESTORS
            - Guidelines Read and Understood: [Yes/No]
            
            SECTION 20: NOTE AND DECLARATION
            - Declaration Signed: [Yes/No]
            
            SECTION 21: APPLICATION CHECKLIST
            - CNIC Copy Attached: [Yes/No]
            - Business/Employment Proof Attached: [Yes/No]
            - Zakat Declaration Attached: [Yes/No]
            - FATCA Form Attached: [Yes/No]
            - CRS Form Attached: [Yes/No]
            - Health Questionnaire Attached: [Yes/No]
            
            SECTION 22: SALES INFORMATION
            - Sales Person's Name: 
            - Manager's Name: 
            - Distributor Stamp: [Present/Absent]
            - DAO Code: 
            - Remarks: 
            
            IMPORTANT: Extract ALL fields exactly as they appear. For checkboxes, write the SELECTED option only.
            Format: Field Name: Value
            """
        },
        "fatca": {
            "name": "FATCA Form – Individual Account",
            "keywords": ["fatca", "us citizen", "us resident", "green card", "tax residence", "place of birth"],
            "prompt": """
            Extract ALL information from the FATCA FORM – INDIVIDUAL ACCOUNT.
            
            SECTION 1: ACCOUNT INFORMATION
            - Title of Account: 
            - CNIC#: 
            - Customer ID: 
            - Country of tax residence: 
            - Place of Birth: 
            
            SECTION 2: US STATUS QUESTIONS
            - US Citizen: [Yes/No]
            - US Resident: [Yes/No]
            - Green Card Holder: [Yes/No]
            - Born in USA: [Yes/No]
            - Transfer funds to USA: [Yes/No]
            - Power of Attorney with US address: [Yes/No]
            - US residence/mailing address: [Yes/No]
            - US telephone number: [Yes/No]
            
            SECTION 3: DECLARATION
            - Signature: 
            - Date: 
            - US Taxpayer ID: 
            
            IMPORTANT: Only extract from FATCA Form.
            Format: Field Name: Value
            """
        },
        "crs": {
            "name": "CRS Form",
            "keywords": ["crs", "tax residence", "tin", "self certification", "common reporting"],
            "prompt": """
            Extract ALL information from the CRS FORM.
            
            SECTION 1: IDENTIFICATION
            - Name as per CNIC: 
            - Father/Husband Name: 
            - CNIC Number: 
            - Date of Birth: 
            - City of Birth: 
            - Country of Birth: 
            
            SECTION 2: ADDRESSES
            - Current Address: 
            - Current Country: 
            - Mailing Address: 
            - Mailing Country: 
            
            SECTION 3: TAX RESIDENCE
            - Tax Residence Country 1: 
            - TIN 1: 
            - Tax Residence Country 2: 
            - TIN 2: 
            - Tax Residence Country 3: 
            - TIN 3: 
            
            SECTION 4: DECLARATION
            - Signature: 
            - Date: 
            
            IMPORTANT: Only extract from CRS Form.
            Format: Field Name: Value
            """
        }
    }
    
    # Initialize data storage
    extracted_data = {
        "account_opening": {},
        "fatca": {},
        "crs": {}
    }
    
    # Process each page
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_groups = len(AL_MEEZAN_PAGE_GROUPS)
    completed = 0

    for form_key, pages in AL_MEEZAN_PAGE_GROUPS.items():
        status_text.text(f"🔍 Processing {form_key.upper()} pages {pages}...")

        # Collect images for this form
        images = []
        for p in pages:
            if p <= len(pdf_images):
                images.append(pdf_images[p - 1])

        if not images:
            continue

        form_config = form_prompts[form_key]

        for idx, page_number in enumerate(pages):

            if page_number > len(pdf_images):
                continue

            page_image = pdf_images[page_number - 1]

            # 🔥 Get page-specific prompt for Account Opening
            if form_key == "account_opening":
                prompt = get_account_opening_prompt_by_page(
                    idx + 1,  # 1, 2, 3
                    form_config["prompt"]
                )
            else:
                prompt = form_config["prompt"]

            status_text.text(
                f"🔍 Processing {form_key.upper()} - Page {page_number}"
            )

            # 🔥 ONE QWEN CALL PER PAGE
            extracted = extract_form_data(
                image=page_image,
                prompt=prompt,
                page_num=str(page_number)
            )
            print('Extracted Information',extracted)
            if extracted:
                parsed = parse_al_meezan_detailed_response(extracted)

                for field, value in parsed.items():
                    if value and value not in ["", "BLANK", "N/A"]:
                        extracted_data[form_key][field] = value

                st.success(
                    f"✅ {form_config['name']} "
                    f"(Page {page_number}) — {len(parsed)} fields extracted"
                )

        completed += 1
        progress_bar.progress(completed / total_groups)

    
    progress_bar.empty()
    status_text.text("✅ Form identification complete!")
    
    # persist extracted data + pdf path + file hash to session so download / rerun doesn't re-run extraction
    st.session_state["al_meezan_extracted_data"] = extracted_data
    if current_hash:
        st.session_state["al_meezan_file_hash"] = current_hash

    try:
        combined = {}
        combined.update(extracted_data.get("account_opening", {}))
        combined.update(extracted_data.get("fatca", {}))
        combined.update(extracted_data.get("crs", {}))
        pdf_path = save_al_meezan_package_to_pdf(combined)
        st.session_state["al_meezan_pdf_path"] = pdf_path
    except Exception:
        pass
    # Display results in tabs
    display_al_meezan_package_tabbed(extracted_data, form_prompts, col2)

def identify_form_on_page(image, form_prompts, page_num):
    """Identify which form is on the current page."""
    try:
        # Use a quick identification prompt
        id_prompt = """
        Look at this page and tell me which form it is:
        
        Options:
        1. INVESTOR ACCOUNT OPENING FORM FOR INDIVIDUAL - Main account details, personal information
        2. FATCA FORM – INDIVIDUAL ACCOUNT - US tax compliance, questions about US citizenship
        3. CRS FORM - Tax residence, TIN numbers, self-certification
        
        Respond with ONLY the form number (1, 2, or 3). If none match, respond with "0".
        """
        
        response = call_openai_api_with_image_single(image, id_prompt, page_num)
        
        if response:
            response = response.strip()
            if "1" in response:
                return "account_opening"
            elif "2" in response:
                return "fatca"
            elif "3" in response:
                return "crs"
        
        return None
        
    except Exception as e:
        st.warning(f"Could not identify form on page {page_num}: {str(e)}")
        return None

def extract_form_data(image, prompt, page_num):
    """Extract data from a specific form."""
    try:
        # Add strict instructions to the prompt
        strict_prompt = f"""
        IMPORTANT: You are looking at a SINGLE FORM on page {page_num}.
        
        {prompt}
        
        STRICT RULES:
        1. Extract ONLY the fields mentioned above
        2. IGNORE all other forms and information
        3. If a field is not found, skip it
        4. Format exactly as: Field Name: Value
        5. No explanations, no extra text
        """
        
        return call_openai_api_with_image_single(image, strict_prompt, page_num)
        
    except Exception as e:
        st.error(f"Error extracting form data from page {page_num}: {str(e)}")
        return None

def parse_al_meezan_detailed_response(response_text):
    """Parse detailed response text for Al Meezan forms."""
    data = {}
    
    if not response_text:
        return data
    
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and section headers
        if not line or line.startswith("SECTION") or line.startswith("IMPORTANT") or line.startswith("STRICT"):
            continue
        
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip()
                
                # Clean up the value
                value = clean_al_meezan_value(value)
                
                if value and value not in ["", "BLANK", "N/A", "Not found", "Not applicable"]:
                    data[field] = value
    
    return data

def clean_al_meezan_value(value):
    """Clean up form field values with enhanced checkbox handling for Al Meezan forms."""
    if not value:
        return ""
    
    # Handle various checkbox formats
    if any(marker in value for marker in ['[✓]', '[X]', '✓', '✔', '☑', '[SELECTED]', '[CHECKED]']):
        # This is a checked checkbox
        if any(marker in value for marker in ['[✓]', '✓', '✔', '[SELECTED]', '[CHECKED]']):
            return "Yes"
        elif '[X]' in value:
            return "No"
    
    # Remove brackets but keep content
    if '[' in value and ']' in value:
        # Extract content inside brackets
        import re
        bracket_content = re.findall(r'\[([^]]+)\]', value)
        if bracket_content:
            # If there's text outside brackets, keep it
            text_outside = re.sub(r'\[[^]]*\]', '', value).strip()
            if text_outside:
                return f"{text_outside} ({bracket_content[0]})"
            else:
                return bracket_content[0].strip()
    
    # Clean up checkboxes
    value = value.replace('✓', 'Yes').replace('✔', 'Yes').replace('[X]', 'No').replace('[ ]', 'No')
    value = value.replace('[', '').replace(']', '').strip()
    
    return value

def display_al_meezan_package_tabbed(extracted_data, form_prompts, col2):
    """Display the 3 targeted forms' results in tabs with the same styling."""
    
    with col2:
        st.markdown("### 📝 Al Meezan Package - 3 Forms Data")

        if any(extracted_data.values()):
            # Combine all extracted data
            all_data = {}
            for form_key in extracted_data:
                all_data.update(extracted_data[form_key])
            
            # Generate PDF immediately
            try:
                pdf_path = save_al_meezan_package_to_pdf(all_data)
                st.session_state["al_meezan_pdf_path"] = pdf_path
                
                # Show download button immediately
                with open(pdf_path, "rb") as f:
                    st.sidebar.download_button(
                        "📥 Download All Forms PDF", 
                        f, 
                        file_name="al_meezan_package_data.pdf", 
                        mime="application/pdf", 
                        use_container_width=True,
                        key="al_meezan_download_btn"
                    )
                    st.success("✅ PDF ready for download!")
            except Exception as e:
                st.error(f"Error creating PDF: {str(e)}")
        
        # Create tabs for each form
        tab_names = [
            "Investor Account Opening Form", 
            "FATCA Form", 
            "CRS Form"
        ]
        
        tab_keys = ["account_opening", "fatca", "crs"]
        
        tabs = st.tabs(tab_names)
        
        # Store edited data
        edited_forms_data = {
            "account_opening": {},
            "fatca": {},
            "crs": {}
        }
        
        for tab_idx, (tab_name, form_key) in enumerate(zip(tab_names, tab_keys)):
            with tabs[tab_idx]:
                form_data = extracted_data[form_key]
                
                if form_data:
                    with st.container(height=700):
                        with st.form(f"{form_key}_form"):
                            edited_form = {}
                            cols = st.columns(2)
                            
                            # Display fields in two columns
                            field_list = list(form_data.items())
                            for idx, (field, value) in enumerate(field_list):
                                col = cols[idx % 2]
                                with col:
                                    edited_form[field] = st.text_input(
                                        field, 
                                        value=value if value else "",
                                        key=f"{form_key}_{field}"
                                    )
                            
                            submitted = st.form_submit_button(f"Save {tab_name} Data")
                            
                            if submitted:
                                edited_forms_data[form_key] = edited_form
                                st.success(f"✅ {tab_name} data saved!")
                    
                    # Show field count
                    st.caption(f"📋 {len(form_data)} fields extracted")
                else:
                    st.info("No data extracted for this form. It might not be in the uploaded document.")
        
        # Combine all edited forms when any is submitted
        all_edited_data = {}
        for form_key in edited_forms_data:
            if edited_forms_data[form_key]:
                all_edited_data.update(edited_forms_data[form_key])
        
        # Save PDF if data exists
        if all_edited_data:
            # Combine original extracted data with edited data
            combined_data = {}
            for form_key in extracted_data:
                combined_data.update(extracted_data[form_key])
            combined_data.update(all_edited_data)
            
            pdf_path = save_al_meezan_package_to_pdf(combined_data)
            
            # Store in session state for download
            st.session_state["al_meezan_edited_data"] = combined_data
            st.session_state["al_meezan_pdf_path"] = pdf_path
        
        # Display extracted and structured data (outside tabs)
        st.markdown("---")
        
        # Show raw extracted data for each form
        for form_key in extracted_data:
            if extracted_data[form_key]:
                st.subheader(f'{form_prompts[form_key]["name"]} - Extracted Data')
                for field, value in extracted_data[form_key].items():
                    st.write(f"{field}: {value}")
        
        # Download button (always visible)
        if "al_meezan_pdf_path" in st.session_state:
            with open(st.session_state["al_meezan_pdf_path"], "rb") as f:
                st.sidebar.download_button(
                    "Download All Forms PDF", 
                    f, 
                    file_name="al_meezan_package_data.pdf", 
                    mime="application/pdf", 
                    use_container_width=True
                )


def parse_al_meezan_package(response_text: str) -> dict:
    """Parse Al Meezan Package - 3 Forms"""
    data = {}
    
    # Form indicator
    data["Form Type"] = "Al Meezan Package - 3 Forms"
    
    # ====================================
    # FORM 1: INVESTOR ACCOUNT OPENING FORM
    # ====================================
    
    # Account Information
    account_info_fields = [
        "Customer ID", "Portfolio No", "Date", "Day", "Month", "Year", 
        "Type of Account"
    ]
    
    # Personal Details
    personal_fields = [
        "Name", "Father's/Husband's Name", "Mother's Maiden Name",
        "CNIC/NICOP/Passport No", "Issuance Date", "Expiry Date",
        "Date of Birth", "Marital Status", "Religion", "Place of Birth",
        "Nationality", "Dual Nationality"
    ]
    
    # Address Information
    address_fields = [
        "Mailing Address", "Mailing City", "Mailing Country",
        "Current Address", "Current City", "Current Country"
    ]
    
    # Contact Details
    contact_fields = [
        "Residential Status", "Email", "Mobile", "Mobile Network",
        "Tel/Res Office"
    ]
    
    # Bank Details
    bank_fields = [
        "Bank Account No", "Bank Name", "Branch", "City"
    ]
    
    # Special Instructions
    special_instruction_fields = [
        "Account Operating Instructions", "Dividend Mandate",
        "Communication Mode", "Stock Dividend"
    ]
    
    # MTPF Details
    mtpf_fields = [
        "Expected Retirement Date", "Note for Pension Fund investments"
    ]
    
    # Allocation Scheme
    allocation_fields = ["Selected Allocation Scheme"]
    
    # Source of Income & Wealth
    income_wealth_fields = [
        "Source of Income", "Source of Wealth", "Name of Employer/Business",
        "Designation", "Nature of Business"
    ]
    
    # Education & Geography
    education_fields = [
        "Education", "Geographies involved", "Type of Counterparties"
    ]
    
    # Transaction Details
    transaction_fields = [
        "Possible Modes of Transactions", "Expected Turnover in Account",
        "Expected Amount of Investment", "Annual Income", "Expected No. of Transactions"
    ]
    
    # Risk Assessment
    risk_fields = [
        "Age Group", "Risk-Return Tolerance", "Monthly Savings", "Occupation"
    ]
    
    # Investment Knowledge
    investment_fields = [
        "Investment Knowledge Level", "Investment Objective", "Investment Horizon"
    ]
    
    # Portfolio Calculation
    portfolio_fields = [
        "Total Score", "Recommended Portfolio", "Calculated ideal Portfolio"
    ]
    
    # Next of Kin
    next_of_kin_fields = [
        "Next of Kin Name", "Next of Kin Contact", "Next of Kin Address"
    ]
    
    # Beneficiary Details
    beneficiary_fields = [
        "Ultimate Beneficiary Name", "Relation with Customer", 
        "Beneficiary CNIC/NICOP/Passport No"
    ]
    
    # Guidelines
    guidelines_fields = ["Guidelines Read and Understood"]
    
    # Declaration
    declaration_fields = ["Declaration Signed"]
    
    # Application Checklist
    checklist_fields = [
        "CNIC Copy Attached", "Business/Employment Proof Attached",
        "Zakat Declaration Attached", "FATCA Form Attached",
        "CRS Form Attached", "Health Questionnaire Attached"
    ]
    
    # Sales Information
    sales_fields = [
        "Sales Person's Name", "Manager's Name", "Distributor Stamp",
        "DAO Code", "Remarks"
    ]
    
    # ====================================
    # FORM 2: FATCA FORM
    # ====================================
    
    # FATCA Account Information
    fatca_account_fields = [
        "FATCA - Title of Account", "FATCA - CNIC#", "FATCA - Customer ID",
        "FATCA - Country of tax residence", "FATCA - Place of Birth"
    ]
    
    # FATCA US Status Questions
    fatca_status_fields = [
        "FATCA - US Citizen", "FATCA - US Resident", "FATCA - Green Card Holder",
        "FATCA - Born in USA", "FATCA - Transfer funds to USA",
        "FATCA - Power of Attorney with US address", 
        "FATCA - US residence/mailing address", "FATCA - US telephone number"
    ]
    
    # FATCA Declaration
    fatca_declaration_fields = [
        "FATCA - Signature", "FATCA - Date", "FATCA - US Taxpayer ID"
    ]
    
    # ====================================
    # FORM 3: CRS FORM
    # ====================================
    
    # CRS Identification
    crs_identification_fields = [
        "CRS - Name as per CNIC", "CRS - Father/Husband Name", 
        "CRS - CNIC Number", "CRS - Date of Birth", 
        "CRS - City of Birth", "CRS - Country of Birth"
    ]
    
    # CRS Addresses
    crs_address_fields = [
        "CRS - Current Address", "CRS - Current Country",
        "CRS - Mailing Address", "CRS - Mailing Country"
    ]
    
    # CRS Tax Residence
    crs_tax_fields = [
        "CRS - Tax Residence Country 1", "CRS - TIN 1",
        "CRS - Tax Residence Country 2", "CRS - TIN 2",
        "CRS - Tax Residence Country 3", "CRS - TIN 3"
    ]
    
    # CRS Declaration
    crs_declaration_fields = ["CRS - Signature", "CRS - Date"]
    
    # Combine all field lists
    all_fields = (
        account_info_fields + personal_fields + address_fields + 
        contact_fields + bank_fields + special_instruction_fields +
        mtpf_fields + allocation_fields + income_wealth_fields +
        education_fields + transaction_fields + risk_fields +
        investment_fields + portfolio_fields + next_of_kin_fields +
        beneficiary_fields + guidelines_fields + declaration_fields +
        checklist_fields + sales_fields + fatca_account_fields +
        fatca_status_fields + fatca_declaration_fields +
        crs_identification_fields + crs_address_fields +
        crs_tax_fields + crs_declaration_fields
    )
    
    # Extract values for all fields
    for field in all_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
        else:
            # Check for alternative field names
            alt_field = field.replace("FATCA - ", "").replace("CRS - ", "")
            alt_value = extract_value(response_text, alt_field)
            if alt_value:
                data[field] = alt_value
            elif "FATCA" in field or "CRS" in field:
                # For FATCA/CRS specific fields, look in their sections
                section_start = response_text.find("FORM 2:") if "FATCA" in field else response_text.find("FORM 3:")
                section_end = response_text.find("FORM 3:") if "FATCA" in field else len(response_text)
                
                if section_start != -1:
                    section_text = response_text[section_start:section_end]
                    section_value = extract_value(section_text, field.replace("FATCA - ", "").replace("CRS - ", ""))
                    if section_value:
                        data[field] = section_value
    
    return data


def save_al_meezan_package_to_pdf(form_data):
    """Save Al Meezan Package (3 forms) data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "al_meezan_package_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "AL MEEZAN PACKAGE - 3 FORMS")
    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 70, "Extracted Form Data")
    
    y = PAGE_HEIGHT - 100
    
    # Form 1: Account Opening Form
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 1: INVESTOR ACCOUNT OPENING FORM FOR INDIVIDUAL")
    y -= 25
    
    # Group fields by form
    account_opening_fields = [
        "Customer ID", "Portfolio No", "Date", "Type of Account",
        "Name", "Father's/Husband's Name", "Mother's Maiden Name",
        "CNIC/NICOP/Passport No", "Issuance Date", "Expiry Date",
        "Date of Birth", "Marital Status", "Religion", "Place of Birth",
        "Nationality", "Dual Nationality", "Mailing Address", "Mailing City",
        "Mailing Country", "Current Address", "Current City", "Current Country",
        "Residential Status", "Email", "Mobile", "Mobile Network",
        "Tel/Res Office", "Bank Account No", "Bank Name", "Branch", "City",
        "Account Operating Instructions", "Dividend Mandate",
        "Communication Mode", "Stock Dividend", "Expected Retirement Date",
        "Selected Allocation Scheme", "Source of Income", "Source of Wealth",
        "Name of Employer/Business", "Designation", "Nature of Business",
        "Education", "Geographies involved", "Type of Counterparties",
        "Possible Modes of Transactions", "Expected Turnover in Account",
        "Expected Amount of Investment", "Annual Income", "Expected No. of Transactions",
        "Age Group", "Risk-Return Tolerance", "Monthly Savings", "Occupation",
        "Investment Knowledge Level", "Investment Objective", "Investment Horizon",
        "Total Score", "Recommended Portfolio", "Calculated ideal Portfolio",
        "Next of Kin Name", "Next of Kin Contact", "Next of Kin Address",
        "Ultimate Beneficiary Name", "Relation with Customer",
        "Beneficiary CNIC/NICOP/Passport No", "Guidelines Read and Understood",
        "Declaration Signed", "CNIC Copy Attached", "Business/Employment Proof Attached",
        "Zakat Declaration Attached", "FATCA Form Attached", "CRS Form Attached",
        "Health Questionnaire Attached", "Sales Person's Name", "Manager's Name",
        "Distributor Stamp", "DAO Code", "Remarks"
    ]
    
    # Display Account Opening Form fields
    c.setFont("Helvetica", 9)
    for field in account_opening_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            # Check for new page
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "ACCOUNT OPENING FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            if len(value) > 80:
                value = value[:77] + "..."
            
            c.drawString(40, y, f"{field}: {value}")
            y -= 15
    
    y -= 20
    
    # Form 2: FATCA Form
    if y < 100:
        c.showPage()
        y = PAGE_HEIGHT - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 2: FATCA FORM – INDIVIDUAL ACCOUNT")
    y -= 25
    
    fatca_fields = [
        "FATCA - Title of Account", "FATCA - CNIC#", "FATCA - Customer ID",
        "FATCA - Country of tax residence", "FATCA - Place of Birth",
        "FATCA - US Citizen", "FATCA - US Resident", "FATCA - Green Card Holder",
        "FATCA - Born in USA", "FATCA - Transfer funds to USA",
        "FATCA - Power of Attorney with US address", 
        "FATCA - US residence/mailing address", "FATCA - US telephone number",
        "FATCA - Signature", "FATCA - Date", "FATCA - US Taxpayer ID"
    ]
    
    c.setFont("Helvetica", 9)
    for field in fatca_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "FATCA FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            c.drawString(40, y, f"{field.replace('FATCA - ', '')}: {value}")
            y -= 15
    
    y -= 20
    
    # Form 3: CRS Form
    if y < 100:
        c.showPage()
        y = PAGE_HEIGHT - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 3: CRS FORM")
    y -= 25
    
    crs_fields = [
        "CRS - Name as per CNIC", "CRS - Father/Husband Name", 
        "CRS - CNIC Number", "CRS - Date of Birth", 
        "CRS - City of Birth", "CRS - Country of Birth",
        "CRS - Current Address", "CRS - Current Country",
        "CRS - Mailing Address", "CRS - Mailing Country",
        "CRS - Tax Residence Country 1", "CRS - TIN 1",
        "CRS - Tax Residence Country 2", "CRS - TIN 2",
        "CRS - Tax Residence Country 3", "CRS - TIN 3",
        "CRS - Signature", "CRS - Date"
    ]
    
    c.setFont("Helvetica", 9)
    for field in crs_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "CRS FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            c.drawString(40, y, f"{field.replace('CRS - ', '')}: {value}")
            y -= 15
    
    c.save()
    return pdf_file



def parse_al_meezan_package(response_text: str) -> dict:
    """Parse Al Meezan Package - 3 Forms"""
    data = {}
    
    # Form indicator
    data["Form Type"] = "Al Meezan Package - 3 Forms"
    
    # ====================================
    # FORM 1: INVESTOR ACCOUNT OPENING FORM
    # ====================================
    
    # Account Information
    account_info_fields = [
        "Customer ID", "Portfolio No", "Date", "Day", "Month", "Year", 
        "Type of Account"
    ]
    
    # Personal Details
    personal_fields = [
        "Name", "Father's/Husband's Name", "Mother's Maiden Name",
        "CNIC/NICOP/Passport No", "Issuance Date", "Expiry Date",
        "Date of Birth", "Marital Status", "Religion", "Place of Birth",
        "Nationality", "Dual Nationality"
    ]
    
    # Address Information
    address_fields = [
        "Mailing Address", "Mailing City", "Mailing Country",
        "Current Address", "Current City", "Current Country"
    ]
    
    # Contact Details
    contact_fields = [
        "Residential Status", "Email", "Mobile", "Mobile Network",
        "Tel/Res Office"
    ]
    
    # Bank Details
    bank_fields = [
        "Bank Account No", "Bank Name", "Branch", "City"
    ]
    
    # Special Instructions
    special_instruction_fields = [
        "Account Operating Instructions", "Dividend Mandate",
        "Communication Mode", "Stock Dividend"
    ]
    
    # MTPF Details
    mtpf_fields = [
        "Expected Retirement Date", "Note for Pension Fund investments"
    ]
    
    # Allocation Scheme
    allocation_fields = ["Selected Allocation Scheme"]
    
    # Source of Income & Wealth
    income_wealth_fields = [
        "Source of Income", "Source of Wealth", "Name of Employer/Business",
        "Designation", "Nature of Business"
    ]
    
    # Education & Geography
    education_fields = [
        "Education", "Geographies involved", "Type of Counterparties"
    ]
    
    # Transaction Details
    transaction_fields = [
        "Possible Modes of Transactions", "Expected Turnover in Account",
        "Expected Amount of Investment", "Annual Income", "Expected No. of Transactions"
    ]
    
    # Risk Assessment
    risk_fields = [
        "Age Group", "Risk-Return Tolerance", "Monthly Savings", "Occupation"
    ]
    
    # Investment Knowledge
    investment_fields = [
        "Investment Knowledge Level", "Investment Objective", "Investment Horizon"
    ]
    
    # Portfolio Calculation
    portfolio_fields = [
        "Total Score", "Recommended Portfolio", "Calculated ideal Portfolio"
    ]
    
    # Next of Kin
    next_of_kin_fields = [
        "Next of Kin Name", "Next of Kin Contact", "Next of Kin Address"
    ]
    
    # Beneficiary Details
    beneficiary_fields = [
        "Ultimate Beneficiary Name", "Relation with Customer", 
        "Beneficiary CNIC/NICOP/Passport No"
    ]
    
    # Guidelines
    guidelines_fields = ["Guidelines Read and Understood"]
    
    # Declaration
    declaration_fields = ["Declaration Signed"]
    
    # Application Checklist
    checklist_fields = [
        "CNIC Copy Attached", "Business/Employment Proof Attached",
        "Zakat Declaration Attached", "FATCA Form Attached",
        "CRS Form Attached", "Health Questionnaire Attached"
    ]
    
    # Sales Information
    sales_fields = [
        "Sales Person's Name", "Manager's Name", "Distributor Stamp",
        "DAO Code", "Remarks"
    ]
    
    # ====================================
    # FORM 2: FATCA FORM
    # ====================================
    
    # FATCA Account Information
    fatca_account_fields = [
        "FATCA - Title of Account", "FATCA - CNIC#", "FATCA - Customer ID",
        "FATCA - Country of tax residence", "FATCA - Place of Birth"
    ]
    
    # FATCA US Status Questions
    fatca_status_fields = [
        "FATCA - US Citizen", "FATCA - US Resident", "FATCA - Green Card Holder",
        "FATCA - Born in USA", "FATCA - Transfer funds to USA",
        "FATCA - Power of Attorney with US address", 
        "FATCA - US residence/mailing address", "FATCA - US telephone number"
    ]
    
    # FATCA Declaration
    fatca_declaration_fields = [
        "FATCA - Signature", "FATCA - Date", "FATCA - US Taxpayer ID"
    ]
    
    # ====================================
    # FORM 3: CRS FORM
    # ====================================
    
    # CRS Identification
    crs_identification_fields = [
        "CRS - Name as per CNIC", "CRS - Father/Husband Name", 
        "CRS - CNIC Number", "CRS - Date of Birth", 
        "CRS - City of Birth", "CRS - Country of Birth"
    ]
    
    # CRS Addresses
    crs_address_fields = [
        "CRS - Current Address", "CRS - Current Country",
        "CRS - Mailing Address", "CRS - Mailing Country"
    ]
    
    # CRS Tax Residence
    crs_tax_fields = [
        "CRS - Tax Residence Country 1", "CRS - TIN 1",
        "CRS - Tax Residence Country 2", "CRS - TIN 2",
        "CRS - Tax Residence Country 3", "CRS - TIN 3"
    ]
    
    # CRS Declaration
    crs_declaration_fields = ["CRS - Signature", "CRS - Date"]
    
    # Combine all field lists
    all_fields = (
        account_info_fields + personal_fields + address_fields + 
        contact_fields + bank_fields + special_instruction_fields +
        mtpf_fields + allocation_fields + income_wealth_fields +
        education_fields + transaction_fields + risk_fields +
        investment_fields + portfolio_fields + next_of_kin_fields +
        beneficiary_fields + guidelines_fields + declaration_fields +
        checklist_fields + sales_fields + fatca_account_fields +
        fatca_status_fields + fatca_declaration_fields +
        crs_identification_fields + crs_address_fields +
        crs_tax_fields + crs_declaration_fields
    )
    
    # Extract values for all fields
    for field in all_fields:
        value = extract_value(response_text, field)
        if value:
            data[field] = value
        else:
            # Check for alternative field names
            alt_field = field.replace("FATCA - ", "").replace("CRS - ", "")
            alt_value = extract_value(response_text, alt_field)
            if alt_value:
                data[field] = alt_value
            elif "FATCA" in field or "CRS" in field:
                # For FATCA/CRS specific fields, look in their sections
                section_start = response_text.find("FORM 2:") if "FATCA" in field else response_text.find("FORM 3:")
                section_end = response_text.find("FORM 3:") if "FATCA" in field else len(response_text)
                
                if section_start != -1:
                    section_text = response_text[section_start:section_end]
                    section_value = extract_value(section_text, field.replace("FATCA - ", "").replace("CRS - ", ""))
                    if section_value:
                        data[field] = section_value
    
    return data


def save_al_meezan_package_to_pdf(form_data):
    """Save Al Meezan Package (3 forms) data to PDF"""
    PAGE_WIDTH, PAGE_HEIGHT = A4
    pdf_file = "al_meezan_package_data.pdf"
    c = canvas.Canvas(pdf_file, pagesize=A4)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 50, "AL MEEZAN PACKAGE - 3 FORMS")
    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 70, "Extracted Form Data")
    
    y = PAGE_HEIGHT - 100
    
    # Form 1: Account Opening Form
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 1: INVESTOR ACCOUNT OPENING FORM FOR INDIVIDUAL")
    y -= 25
    
    # Group fields by form
    account_opening_fields = [
        "Customer ID", "Portfolio No", "Date", "Type of Account",
        "Name", "Father's/Husband's Name", "Mother's Maiden Name",
        "CNIC/NICOP/Passport No", "Issuance Date", "Expiry Date",
        "Date of Birth", "Marital Status", "Religion", "Place of Birth",
        "Nationality", "Dual Nationality", "Mailing Address", "Mailing City",
        "Mailing Country", "Current Address", "Current City", "Current Country",
        "Residential Status", "Email", "Mobile", "Mobile Network",
        "Tel/Res Office", "Bank Account No", "Bank Name", "Branch", "City",
        "Account Operating Instructions", "Dividend Mandate",
        "Communication Mode", "Stock Dividend", "Expected Retirement Date",
        "Selected Allocation Scheme", "Source of Income", "Source of Wealth",
        "Name of Employer/Business", "Designation", "Nature of Business",
        "Education", "Geographies involved", "Type of Counterparties",
        "Possible Modes of Transactions", "Expected Turnover in Account",
        "Expected Amount of Investment", "Annual Income", "Expected No. of Transactions",
        "Age Group", "Risk-Return Tolerance", "Monthly Savings", "Occupation",
        "Investment Knowledge Level", "Investment Objective", "Investment Horizon",
        "Total Score", "Recommended Portfolio", "Calculated ideal Portfolio",
        "Next of Kin Name", "Next of Kin Contact", "Next of Kin Address",
        "Ultimate Beneficiary Name", "Relation with Customer",
        "Beneficiary CNIC/NICOP/Passport No", "Guidelines Read and Understood",
        "Declaration Signed", "CNIC Copy Attached", "Business/Employment Proof Attached",
        "Zakat Declaration Attached", "FATCA Form Attached", "CRS Form Attached",
        "Health Questionnaire Attached", "Sales Person's Name", "Manager's Name",
        "Distributor Stamp", "DAO Code", "Remarks"
    ]
    
    # Display Account Opening Form fields
    c.setFont("Helvetica", 9)
    for field in account_opening_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            # Check for new page
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "ACCOUNT OPENING FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            if len(value) > 80:
                value = value[:77] + "..."
            
            c.drawString(40, y, f"{field}: {value}")
            y -= 15
    
    y -= 20
    
    # Form 2: FATCA Form
    if y < 100:
        c.showPage()
        y = PAGE_HEIGHT - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 2: FATCA FORM – INDIVIDUAL ACCOUNT")
    y -= 25
    
    fatca_fields = [
        "FATCA - Title of Account", "FATCA - CNIC#", "FATCA - Customer ID",
        "FATCA - Country of tax residence", "FATCA - Place of Birth",
        "FATCA - US Citizen", "FATCA - US Resident", "FATCA - Green Card Holder",
        "FATCA - Born in USA", "FATCA - Transfer funds to USA",
        "FATCA - Power of Attorney with US address", 
        "FATCA - US residence/mailing address", "FATCA - US telephone number",
        "FATCA - Signature", "FATCA - Date", "FATCA - US Taxpayer ID"
    ]
    
    c.setFont("Helvetica", 9)
    for field in fatca_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "FATCA FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            c.drawString(40, y, f"{field.replace('FATCA - ', '')}: {value}")
            y -= 15
    
    y -= 20
    
    # Form 3: CRS Form
    if y < 100:
        c.showPage()
        y = PAGE_HEIGHT - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "FORM 3: CRS FORM")
    y -= 25
    
    crs_fields = [
        "CRS - Name as per CNIC", "CRS - Father/Husband Name", 
        "CRS - CNIC Number", "CRS - Date of Birth", 
        "CRS - City of Birth", "CRS - Country of Birth",
        "CRS - Current Address", "CRS - Current Country",
        "CRS - Mailing Address", "CRS - Mailing Country",
        "CRS - Tax Residence Country 1", "CRS - TIN 1",
        "CRS - Tax Residence Country 2", "CRS - TIN 2",
        "CRS - Tax Residence Country 3", "CRS - TIN 3",
        "CRS - Signature", "CRS - Date"
    ]
    
    c.setFont("Helvetica", 9)
    for field in crs_fields:
        if field in form_data and form_data[field] not in ["", "Not provided"]:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, PAGE_HEIGHT - 50, "CRS FORM (Continued)")
                y = PAGE_HEIGHT - 80
                c.setFont("Helvetica", 9)
            
            value = str(form_data[field])
            c.drawString(40, y, f"{field.replace('CRS - ', '')}: {value}")
            y -= 15
    
    c.save()
    return pdf_file

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Multi-Bank Form Parser", layout="wide")

# ---------- BRAND COLORS ----------
PRIMARY_COLOR = "#1E1E1E"     # Deep Dark Gray/Black (Main background)
SECONDARY_COLOR = "#00B894"   # Mint/Teal (Logo/Heading accent)
BACKGROUND_COLOR = "#F8F9FA"

# ---------- LOGO PATHS ----------
logo_path = "OfficeFlow Ai-01-01.png"
dashboard_preview = "dashboard.png"

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
        height: 105vh;
        width: 100%;
        color: white;
        padding: 3rem 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50px 50px 50px 50px;
        overflow: hidden;
    }}
    .left-side img {{
        width: 110%;
        max-width: 700px;
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
    .stTextInput > div > div > div > input {{
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
    st.title("Multi-Bank Form Parser")
   
    # Concise app description
    st.info("""
    **AI-Powered Bank Form Processing**
    
    Automatically extracts data from scanned banking forms with advanced OCR and AI technology. 
    Supports multiple Pakistani banks including Meezan, MCB, Allied, Alfalah, and Askari Bank.
    
    Upload your form (JPG, JPEG, PNG, or PDF) to get started!
    """)

    st.sidebar.image('OfficeFlow Ai-01-01.png', use_container_width='auto')
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))
    
    # Multi-bank dropdown with ALL 5 banks
    bank_option = st.sidebar.selectbox(
        "Select Bank Form Type:",
        ["Meezan Bank", "Al Meezan Multi Page Form","MCB Bank", "Allied Bank", "Alfalah Bank", "Askari Bank", "MCB Redemption C-1", "MCB Early Redemption","Custom Form"],
        key="bank_selector"
    )
    
    uploaded_file = st.sidebar.file_uploader("📤 Upload scanned form", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file:
        new_hash = file_hash(uploaded_file)
        if "file_hash" not in st.session_state or st.session_state.file_hash != new_hash:
            st.session_state.file_hash = new_hash
            # Clear previous responses when file changes
            for key in ["meezan_response1", "meezan_response2", "meezan_response3", 
                       "mcb_response", "allied_response", "alfalah_response", 
                       "askari_response", "mcb_redemption_c1_response", "mcb_early_redemption_response", "al_meezan_package_response","custom_response", "al_meezan_extracted_data", "al_meezan_pdf_path", "al_meezan_edited_data", "al_meezan_file_hash"]:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Display the uploaded file
        if uploaded_file.type == "application/pdf":
            # For PDF files, convert to image and display first page
            if PDF_SUPPORT:
                pdf_images = convert_pdf_to_images(uploaded_file)
                if pdf_images and len(pdf_images) > 0:
                    image_pil = pdf_images[0]
                    st.info(f"📄 PDF detected with {len(pdf_images)} pages. Processing first page.")
                else:
                    st.error("Failed to convert PDF to images.")
        else:
            # For image files
            image_pil = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_pil, caption=f"Uploaded {bank_option} Form", use_container_width=True)
        
        # Process based on selected bank
        if bank_option == "Meezan Bank":
            process_meezan_form(uploaded_file, col2)
        elif bank_option == "Al Meezan Multi Page Form":
            process_al_meezan_package(uploaded_file, col2)
        elif bank_option == "MCB Bank":
            process_mcb_form(uploaded_file, col2)
        elif bank_option == "Allied Bank":
            process_allied_form(uploaded_file, col2)
        elif bank_option == "Alfalah Bank":
            process_alfalah_form(uploaded_file, col2)
        elif bank_option == "Askari Bank":
            process_askari_form(uploaded_file, col2)
        elif bank_option == "MCB Redemption C-1":  
            process_mcb_redemption_c1(uploaded_file, col2)  
        elif bank_option == "MCB Early Redemption":
            process_mcb_early_redemption(uploaded_file, col2)
        else:
            # Custom form processing
            if "custom_response" not in st.session_state:
                with st.spinner("Extracting data from custom form..."):
                    response = call_openai_api_with_image(uploaded_file)
                    st.session_state.custom_response = response
            
            with col2:
                st.subheader("Extracted Data from Custom Form")
                st.write(st.session_state.custom_response)

else:
    # Login page
    col1, col2 = st.columns([1.5, 1])

    with col1:
        img_path = Path(dashboard_preview)
        if img_path.exists():
            img_bytes = img_path.read_bytes()
            mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            html = f'<div class="left-side"><img src="data:{mime};base64,{b64}" alt="dashboard preview"/></div>'
        else:
            html = f'<div class="left-side"><div style="color:#fff">Dashboard image not found: {dashboard_preview}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.image("OfficeFlow Ai-01-01.png", width=250)
        st.markdown("<h2>Officeflow AI</h2>", unsafe_allow_html=True)
        
        st.subheader("Login")
        st.write("Welcome! Please enter your credentials to proceed.")
        
        email = st.text_input("Email Address", placeholder="Enter your email")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        
        login = st.button("Login", type="primary")
        
        if login:
            if email == "admin@officeflowai.com" and password == "12345":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
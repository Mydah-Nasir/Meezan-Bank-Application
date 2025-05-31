import streamlit as st
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import openai
import base64
import tempfile
import json
import os
from openai import OpenAI
import re
import numpy as np
import pandas as pd
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from paddleocr import PaddleOCR
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.set_page_config(layout="wide")
st.title("📄 Form Segmentation & Extraction")

# --- Helper Functions ---

def find_heading_y(image_gray, heading_text):
    ocr_data = pytesseract.image_to_data(image_gray, output_type=Output.DICT)
    for i, word in enumerate(ocr_data['text']):
        if heading_text.lower() in word.lower():
            return ocr_data['top'][i]
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

# Initialize OCR once
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def apply_ocr_and_generate_pdfs(segment_paths):
    pdf_paths = []

    for image_path in segment_paths:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        results = ocr.predict(image_path)
        res = results[0]  # Only one image at a time

        texts = res['rec_texts']
        scores = res['rec_scores']
        boxes = res['rec_polys']  # [N, 4, 2]

        # Prepare PDF path
        pdf_path = image_path.replace(".jpg", ".pdf")
        c = canvas.Canvas(pdf_path, pagesize=(width, height))

        for text, score, box in zip(texts, scores, boxes):
            if not text.strip():
                continue

            x, y = box[0]  # top-left of the quadrilateral box
            flipped_y = height - y
            box_height = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
            font_size = max(6, int(box_height * 0.8))

            c.setFont("Helvetica", font_size)
            c.drawString(x, flipped_y, text)

        c.save()
        print(f"OCR text written to: {pdf_path}")
        pdf_paths.append(pdf_path)

    return pdf_paths

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
        3: """In allocation scheme add all that are ticked. Leave blank if none is ticked. Extract this structure only:
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
OPENAI_API_KEY = st.secrets["API_KEY"]


uploaded_file = st.file_uploader("📤 Upload scanned form", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file and OPENAI_API_KEY:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Form", use_container_width=True)

    with st.spinner("Processing image..."):
        segments = segment_image(image_pil)
        pdf_paths = apply_ocr_and_generate_pdfs(segments)
        print("Generated PDF paths:", pdf_paths)

    # all_data = {}
    # for idx, seg_path in enumerate(segments, start=1):
    #     data = call_gpt_vision(seg_path, idx, OPENAI_API_KEY)

    #     if "error" in data:
    #         st.error(data["error"])
    #         st.text(data["raw"])
    #     else:
    #         all_data.update(data)

    # st.success("✅ All segments processed.")
    # flat_data = flatten_dict(all_data)
    # df = pd.DataFrame(list(flat_data.items()), columns=["Field", "Value"])
    # st.dataframe(df, use_container_width=True)

elif uploaded_file:
    st.warning("Please enter your credentials.")

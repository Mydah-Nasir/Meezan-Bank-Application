import streamlit as st
from google import genai
import json
import os
import base64
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import io

load_dotenv()

st.set_page_config(page_title="Al Meezan Form Extractor", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #00A651;
        margin-bottom: 30px;
    }
    .pakistan-flag {
        background: linear-gradient(90deg, #115740 0%, #115740 33%, white 33%, white 66%, #115740 66%, #115740 100%);
        padding: 10px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .form-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #00A651;
    }
    .section-header {
        color: #1E3A8A;
        border-bottom: 2px solid #00A651;
        padding-bottom: 8px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    .field-container {
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #00A651;
    }
    .field-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .field-value {
        font-size: 1rem;
        font-weight: 500;
        color: #1E3A8A;
        word-break: break-word;
        margin-top: 5px;
    }
    .empty-field {
        color: #999;
        font-style: italic;
    }
    .true-value {
        color: #00A651;
        font-weight: 600;
    }
    .false-value {
        color: #dc3545;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #00A651;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background-color: #F8FFF8;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="pakistan-flag">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Al Meezan Investor Account Opening Form Extractor</h1>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
**Extract ALL 350+ fields from Al Meezan Investment Form**
Upload an image or PDF and get structured data page by page.
""")

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    with st.sidebar:
        GEMINI_API_KEY = st.text_input("Google Gemini API Key", type="password")
        if GEMINI_API_KEY:
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

with st.sidebar:
    st.header("Settings")
    st.markdown("---")
    st.markdown("### Form Structure (5 Pages)")
    st.markdown("""
    - **Page 1:** Account Type, Principal Holder, Contact, Minor, Bank, Joint Holders
    - **Page 2:** KYC Details, PEP, Risk Profile, Next of Kin, Beneficiary, Declaration, Office Use
    - **Page 3:** Checklist Documents
    - **Page 4:** FATCA Form
    - **Page 5:** CRS Tax Residency Form
    """)

def pdf_to_images(pdf_file):
    try:
        import fitz
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        total_pages = len(doc)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for page_num in range(total_pages):
            status_text.text(f"Converting page {page_num + 1} of {total_pages}...")
            page = doc[page_num]
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            images.append(img_bytes)
            progress_bar.progress((page_num + 1) / total_pages)
        doc.close()
        status_text.empty()
        progress_bar.empty()
        return images, total_pages
    except ImportError:
        st.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return None, 0
    except Exception as e:
        st.error(f"PDF conversion error: {str(e)}")
        return None, 0

PAGE_1_PROMPT = """You are extracting data from PAGE 1 of Al Meezan Investor Account Opening Form.

Return ONLY valid JSON with this structure:

{
  "type_of_account": {"single": false, "joint": false, "minor": false, "mtpf": false},
  "principal_holder": {
    "title": "", "full_name": "", "father_husband_name": "", "mother_maiden_name": "",
    "cnic_nicop_passport": "", "issuance_date": "", "expiry_date": "",
    "marital_status": {"single": false, "married": false}, 
    "religion": {"muslim": false, "non_muslim": false}, 
    "place_of_birth": "", "date_of_birth": "",
    "nationality": "", "dual_nationality": {"no": false, "yes": false}, 
    "dual_nationality_specify": "",
    "mailing_address": "", "mailing_city": "", "mailing_country": "", 
    "current_address": "", "current_city": "", "current_country": "",
    "residential_status": {"Pakistan_Resident": false, "Non_Resident": false, "Resident_Foreign_National": false, "Non_Resident_Foreign_National": false}, 
    "email": "", "tel_res_office": "", "mobile": "", "mobile_network": ""
  },
  "minor_account": {"guardian_name": "", "relation_with_principal": "", "guardian_cnic": "", "guardian_cnic_expiry_date": ""},
  "bank_account_details": {"iban": "", "bank_name": "", "branch": "", "city": ""},
  "joint_holders": [
    {"name": "", "relation": "", "customer_id": "", "cnic": "", "issuance_date": "", "expiry_date": ""}
  ],
  "account_operating_instruction": {
    "principal_only": false, "either_or_survivor": false, "any_two": false, "all": false
  },
  "dividend_mandate": {
    "cash_dividend_reinvest": false, "cash_dividend_provide_cash": false,
    "stock_dividend_issue_bonus": false, "stock_dividend_encash_bonus": false
  },
  "communication_mode": {"physical_communication": false},
  "mtpf_details": {
    "expected_retirement_date": "",
    "allocation_scheme": {
      "high_volatility": false, "high_volatility_with_gold": false, "variable_volatility": false,
      "medium_volatility": false, "medium_volatility_with_gold": false, "100%_debt": false,
      "low_volatility": false, "low_volatility_with_gold": false, "100%_equity": false,
      "lower_volatility": false, "100%_money_market": false, "lifecycle_plan": false, "100%_gold": false
    }
  },
  "principal_account_holder_signed": false,
  "joint_account_holder_1_signed": false,
  "joint_account_holder_2_signed": false
}"""

PAGE_2_PROMPT = """You are extracting data from PAGE 2 of Al Meezan Investor Account Opening Form.

Return ONLY valid JSON with this structure:

{
  "kyc_details": {
    "source_of_income_business_self_employed": false, 
    "source_of_income_salary": false, 
    "source_of_income_pension": false,
    "source_of_income_rent": false, 
    "source_of_income_profit_dividend": false, 
    "source_of_income_other": false,
    "source_of_wealth_inheritance": false, 
    "source_of_wealth_remittances": false, 
    "source_of_wealth_savings": false,
    "source_of_wealth_stocks_investment": false, 
    "source_of_wealth_other": false,
    "employer_business_name": "", 
    "designation": "", 
    "nature_of_business": "",
    "education_undergraduate": false, 
    "education_graduate": false, 
    "education_postgraduate": false,
    "education_professional": false, 
    "education_other": false,
    "geographies_domestic_sindh": false, 
    "geographies_domestic_punjab": false, 
    "geographies_domestic_kpk": false,
    "geographies_domestic_balochistan": false, 
    "geographies_domestic_other": false,
    "geographies_international_fatf_compliant": false, 
    "geographies_international_fatf_non_compliant": false,
    "modes_of_transactions_online": false, 
    "modes_of_transactions_physical": false, 
    "modes_of_transactions_both": false,
    "expected_transactions_monthly": "", 
    "expected_turnover_monthly": "", 
    "expected_turnover_annually": "",
    "expected_amount_investment": {
      "upto_Rs_2.5M": false, "Rs_2.5M_5M": false, "Rs_5M_10M": false, "above_Rs_10M": false
    }, 
    "annual_income": {
      "upto_Rs_1M": false, "Rs_1M_3M": false, "Rs_3M_6M": false, "Rs_6M_8M": false,
      "Rs_8M_10M": false, "above_Rs_10M": false
    }
  },
  "additional_declarations": {
    "principal_holder": {
      "refused_account_before": false,
      "financially_dependent": false,
      "deals_in_high_value_items": false,
      "high_risk_cash_incentive": false,
      "links_with_offshore_tax_haven_countries": false
    },
    "joint_holder_1": {
      "refused_account_before": false,
      "financially_dependent": false,
      "deals_in_high_value_items": false,
      "high_risk_cash_incentive": false,
      "links_with_offshore_tax_haven_countries": false
    },
    "joint_holder_2": {
      "refused_account_before": false,
      "financially_dependent": false,
      "deals_in_high_value_items": false,
      "high_risk_cash_incentive": false,
      "links_with_offshore_tax_haven_countries": false
    }
  },
  "politically_exposed_person": {
    "principal_holder": {
      "no": false,
      "yes": false
    },
    "joint_holder_1": {
      "no": false,
      "yes": false
    },
    "joint_holder_2": {
      "no": false,
      "yes": false
    }
  },
  "risk_profile_age": {
    "above_60": false,
    "50_60": false,
    "40_50": false,
    "below_40": false
  },
  "risk_profile_tolerance": {
    "lower_risk_lower_returns": false,
    "medium_risk_medium_returns": false,
    "higher_risk_higher_returns": false
  },
  "risk_profile_savings": {
    "1000_25000": false,
    "25000_50000": false,
    "above_50000": false
  },
  "risk_profile_occupation": {
    "retired": false,
    "housewife_student": false,
    "salaried": false,
    "self_employed_business": false
  },
  "risk_profile_objective": {
    "cash_management": false,
    "monthly_income": false,
    "capital_growth": false
  },
  "risk_profile_knowledge": {
    "limited_basic_average": false,
    "good_excellent": false
  },
  "risk_profile_horizon": {
    "less_than_6_months": false,
    "6_months_to_1_year": false,
    "1_to_3_years": false,
    "more_than_3_years": false
  },
  "risk_profile_result": {
    "total_score": null,
    "investor_portfolio": "",
    "recommended_fund": ""
  },
  "next_of_kin": {
    "name": "", 
    "contact_number": "", 
    "relation_with_customer": "", 
    "address": ""
  },
  "beneficiary_details": {
    "ultimate_beneficiary_name": "", 
    "relation_with_customer": "", 
    "cnic_nicop_passport": ""
  },
  "declaration": {
    "principal_signature_present": false, 
    "joint_1_signature_present": false, 
    "joint_2_signature_present": false,
    "read_and_understood_guidelines": false
  },
  "office_use": {
    "sales_person_name": "", 
    "dao_code": "", 
    "manager_name": "", 
    "reporting_date": "", 
    "remarks": ""
  }
}"""

PAGE_3_PROMPT = """You are extracting data from PAGE 3 of Al Meezan Investor Account Opening Form.

Return ONLY valid JSON with this structure:

{
  "checklist_documents": {
    "individual": false, 
    "cnic_copy": false, 
    "business_employment_proof": false,
    "zakat_declaration": false, 
    "crs": false, 
    "health_questionnaire": false, 
    "fatca_form": false
  }
}"""

PAGE_4_PROMPT = """You are extracting data from PAGE 4 (FATCA Form).

Return ONLY valid JSON with this structure:

{
  "fatca_information": {
    "title_of_account": "",
    "cnic": "",
    "customer_id": "",
    "country_of_tax_residence_other_than_pakistan": {"None": false, "USA": false, "Other": false},
    "place_of_birth": "", 
    "city": "", 
    "state": "",
    "us_citizen": false, 
    "us_resident": false, 
    "us_green_card": false,
    "born_in_usa": false, 
    "us_standing_instructions": false,
    "us_power_of_attorney": false, 
    "us_mailing_address": false, 
    "us_telephone": false,
    "section_b_non_us_certification": false,
    "declaration_signed": false,
    "us_tin": ""
  }
}"""

PAGE_5_PROMPT = """You are extracting data from PAGE 5 of Al Meezan Investor Account Opening Form - CRS Tax Residency Form.

Return ONLY valid JSON with this structure:

{
  "crs_information": {
    "personal_information": {
      "name_of_investor": "",
      "father_husband_name": "",
      "cnic": "",
      "date_of_birth": "",
      "place_of_birth_country": "",
      "current_residence_address": "",
      "current_residence_country": "",
      "mailing_address": "",
      "mailing_address_country": ""
    },
    "tax_residency_table": [
      {"row_number": 1, "country_of_tax_residence": "", "tin_or_equivalent": ""},
      {"row_number": 2, "country_of_tax_residence": "", "tin_or_equivalent": ""},
      {"row_number": 3, "country_of_tax_residence": "", "tin_or_equivalent": ""}
    ],
    "declaration": {"i_declare_information_true": false, "date": "", "signature": false}
  }
}"""

PAGE_PROMPTS = {
    1: PAGE_1_PROMPT,
    2: PAGE_2_PROMPT,
    3: PAGE_3_PROMPT,
    4: PAGE_4_PROMPT,
    5: PAGE_5_PROMPT,
}

def extract_page_with_gemini(image_bytes, page_num, model_name):
    client = genai.Client(api_key=GEMINI_API_KEY)
    image = Image.open(io.BytesIO(image_bytes))
    prompt = PAGE_PROMPTS.get(page_num, PAGE_1_PROMPT)
    full_prompt = f"{prompt}\n\nThis is PAGE {page_num} of the form. Extract ONLY what you see on this specific page."
    
    response = client.models.generate_content(
        model=model_name,
        contents=[full_prompt, image]
    )
    
    response_text = response.text
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    return json.loads(response_text)

def extract_all_pages(image_bytes_list, model_name, progress_callback=None):
    all_results = {}
    for idx, img_bytes in enumerate(image_bytes_list):
        page_num = idx + 1
        if page_num > 5:
            break
        if progress_callback:
            progress_callback(page_num, min(len(image_bytes_list), 5))
        try:
            page_result = extract_page_with_gemini(img_bytes, page_num, model_name)
            all_results[page_num] = page_result
        except Exception as e:
            all_results[page_num] = {"error": str(e)}
    return all_results

def display_all_fields(data, prefix=""):
    """Recursively display ALL fields including empty ones and False values"""
    items = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                items.extend(display_all_fields(value, f"{prefix}{key}_"))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(display_all_fields(item, f"{prefix}{key}_{i+1}_"))
                    else:
                        label = f"{prefix}{key}_{i+1}".replace("_", " ").title()
                        items.append((label, item))
            else:
                label = f"{prefix}{key}".replace("_", " ").title()
                items.append((label, value))
    
    return items

def display_additional_declarations(data):
    """Display additional declarations for Principal, Joint Holder 1, and Joint Holder 2"""
    
    if "additional_declarations" not in data:
        return
    
    declarations = data["additional_declarations"]
    
    st.markdown(f'<h3 class="section-header">Additional Declarations</h3>', unsafe_allow_html=True)
    
    fields = [
        ("refused_account_before", "Refused Account Before"),
        ("financially_dependent", "Financially Dependent"),
        ("deals_in_high_value_items", "Deals in High Value Items"),
        ("high_risk_cash_incentive", "High Risk Cash Incentive"),
        ("links_with_offshore_tax_haven_countries", "Links with Offshore Tax Haven Countries")
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Principal Holder**")
        if "principal_holder" in declarations:
            for field_key, field_label in fields:
                value = declarations["principal_holder"].get(field_key, False)
                display_value = "YES" if value else "NO"
                value_class = "true-value" if value else "false-value"
                st.markdown(f"""
                <div class="field-container">
                    <div class="field-label">{field_label}</div>
                    <div class="field-value {value_class}">{display_value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Joint Holder 1**")
        if "joint_holder_1" in declarations:
            for field_key, field_label in fields:
                value = declarations["joint_holder_1"].get(field_key, False)
                display_value = "YES" if value else "NO"
                value_class = "true-value" if value else "false-value"
                st.markdown(f"""
                <div class="field-container">
                    <div class="field-label">{field_label}</div>
                    <div class="field-value {value_class}">{display_value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Joint Holder 2**")
        if "joint_holder_2" in declarations:
            for field_key, field_label in fields:
                value = declarations["joint_holder_2"].get(field_key, False)
                display_value = "YES" if value else "NO"
                value_class = "true-value" if value else "false-value"
                st.markdown(f"""
                <div class="field-container">
                    <div class="field-label">{field_label}</div>
                    <div class="field-value {value_class}">{display_value}</div>
                </div>
                """, unsafe_allow_html=True)

def display_pep_section(data):
    """Display Politically Exposed Person section for all three holders"""
    
    if "politically_exposed_person" not in data:
        return
    
    pep = data["politically_exposed_person"]
    
    st.markdown(f'<h3 class="section-header">Politically Exposed Person (PEP)</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Principal Holder
    with col1:
        st.markdown("**Principal Holder**")
        if "principal_holder" in pep:
            value_yes = pep["principal_holder"].get("yes", False)
            value_no = pep["principal_holder"].get("no", False)
            
            display_yes = "YES" if value_yes else "NO"
            display_no = "YES" if value_no else "NO"
            yes_class = "true-value" if value_yes else "false-value"
            no_class = "true-value" if value_no else "false-value"
            
            st.markdown(f"""
            <div class="field-container">
                <div class="field-label">Is a PEP</div>
                <div class="field-value {yes_class}">{display_yes}</div>
            </div>
            <div class="field-container">
                <div class="field-label">Not a PEP</div>
                <div class="field-value {no_class}">{display_no}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Joint Holder 1
    with col2:
        st.markdown("**Joint Holder 1**")
        if "joint_holder_1" in pep:
            value_yes = pep["joint_holder_1"].get("yes", False)
            value_no = pep["joint_holder_1"].get("no", False)
            
            display_yes = "YES" if value_yes else "NO"
            display_no = "YES" if value_no else "NO"
            yes_class = "true-value" if value_yes else "false-value"
            no_class = "true-value" if value_no else "false-value"
            
            st.markdown(f"""
            <div class="field-container">
                <div class="field-label">Is a PEP</div>
                <div class="field-value {yes_class}">{display_yes}</div>
            </div>
            <div class="field-container">
                <div class="field-label">Not a PEP</div>
                <div class="field-value {no_class}">{display_no}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Joint Holder 2
    with col3:
        st.markdown("**Joint Holder 2**")
        if "joint_holder_2" in pep:
            value_yes = pep["joint_holder_2"].get("yes", False)
            value_no = pep["joint_holder_2"].get("no", False)
            
            display_yes = "YES" if value_yes else "NO"
            display_no = "YES" if value_no else "NO"
            yes_class = "true-value" if value_yes else "false-value"
            no_class = "true-value" if value_no else "false-value"
            
            st.markdown(f"""
            <div class="field-container">
                <div class="field-label">Is a PEP</div>
                <div class="field-value {yes_class}">{display_yes}</div>
            </div>
            <div class="field-container">
                <div class="field-label">Not a PEP</div>
                <div class="field-value {no_class}">{display_no}</div>
            </div>
            """, unsafe_allow_html=True)

def display_risk_profile_section(data):
    """Display risk profile with proper field names"""
    
    # Define the 7 fields and their display names
    risk_fields = {
        "risk_profile_age": "Age (in years)",
        "risk_profile_tolerance": "Risk-Return Tolerance Level",
        "risk_profile_savings": "Monthly Savings",
        "risk_profile_occupation": "Occupation",
        "risk_profile_objective": "Investment Objective",
        "risk_profile_knowledge": "Level of knowledge of Investments and Financial markets",
        "risk_profile_horizon": "Investment Horizon",
        "risk_profile_result": "Results"
    }
    
    # Define option labels for each field
    option_labels = {
        "risk_profile_age": {
            "above_60": "Above 60",
            "50_60": "50-60",
            "40_50": "40-50",
            "below_40": "Below 40"
        },
        "risk_profile_tolerance": {
            "lower_risk_lower_returns": "Lower Risk, Lower Returns",
            "medium_risk_medium_returns": "Medium Risk, Medium Returns",
            "higher_risk_higher_returns": "Higher Risk, Higher Returns"
        },
        "risk_profile_savings": {
            "1000_25000": "Rs. 1,000 - Rs. 25,000",
            "25000_50000": "Rs. 25,000 - Rs. 50,000",
            "above_50000": "Above Rs. 50,000"
        },
        "risk_profile_occupation": {
            "retired": "Retired",
            "housewife_student": "Housewife/Student",
            "salaried": "Salaried",
            "self_employed_business": "Self Employed / Business"
        },
        "risk_profile_objective": {
            "cash_management": "Cash Management",
            "monthly_income": "Monthly Income",
            "capital_growth": "Capital Growth/Long Term Savings/Retirement"
        },
        "risk_profile_knowledge": {
            "limited_basic_average": "Limited/Basic/Average",
            "good_excellent": "Good/Excellent"
        },
        "risk_profile_horizon": {
            "less_than_6_months": "Less than 6 months",
            "6_months_to_1_year": "6 months to 1 year",
            "1_to_3_years": "1 to 3 years",
            "more_than_3_years": "More than 3 years"
        }
    }
    
    # Display each risk field
    for field_key, display_name in risk_fields.items():
        if field_key in data and data[field_key]:
            field_data = data[field_key]
            
            if isinstance(field_data, dict):
                # Check if it has any values
                if field_key == "risk_profile_result":
                    has_value = any(v for v in field_data.values() if v)
                else:
                    has_value = any(v for v in field_data.values() if v)
                
                if not has_value:
                    continue
                
                st.markdown(f'<h3 class="section-header">{display_name}</h3>', unsafe_allow_html=True)
                
                cols = st.columns(2)
                col_idx = 0
                
                if field_key == "risk_profile_result":
                    # Display result fields
                    for key, value in field_data.items():
                        if value and value != "" and value is not None:
                            label = key.replace("_", " ").title()
                            with cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="field-container">
                                    <div class="field-label">{label}</div>
                                    <div class="field-value">{value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            col_idx += 1
                else:
                    # Display checkbox options
                    labels = option_labels.get(field_key, {})
                    for option_key, option_value in field_data.items():
                        if option_key in labels:
                            label = labels[option_key]
                            display_value = "YES" if option_value else "NO"
                            value_class = "true-value" if option_value else "false-value"
                            
                            with cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="field-container">
                                    <div class="field-label">{label}</div>
                                    <div class="field-value {value_class}">{display_value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            col_idx += 1

def display_page_data(page_num, data):
    """Display ALL data for a page"""
    with st.container():
        st.markdown(f'<div class="form-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: #00A651;">Page {page_num}</h2>', unsafe_allow_html=True)
        
        if "error" in data:
            st.error(f"Error on this page: {data['error']}")
        else:
            # Handle Page 2 specially
            if page_num == 2:
                # Display KYC Details
                if "kyc_details" in data:
                    st.markdown(f'<h3 class="section-header">KYC Details</h3>', unsafe_allow_html=True)
                    all_fields = display_all_fields(data["kyc_details"])
                    if all_fields:
                        cols = st.columns(2)
                        col_idx = 0
                        for label, value in all_fields:
                            if isinstance(value, bool):
                                display_value = "YES" if value else "NO"
                                value_class = "true-value" if value else "false-value"
                            elif value == "" or value is None:
                                display_value = "(empty)"
                                value_class = "empty-field"
                            else:
                                display_value = str(value)
                                value_class = ""
                            with cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="field-container">
                                    <div class="field-label">{label}</div>
                                    <div class="field-value {value_class}">{display_value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            col_idx += 1
                
                # Display Additional Declarations
                display_additional_declarations(data)
                
                # Display PEP section
                display_pep_section(data)
                
                # Display risk profile sections
                display_risk_profile_section(data)
                
                # Display other sections
                other_sections = ["next_of_kin", "beneficiary_details", "declaration", "office_use"]
                for section_name in other_sections:
                    if section_name in data and data[section_name]:
                        st.markdown(f'<h3 class="section-header">{section_name.replace("_", " ").title()}</h3>', unsafe_allow_html=True)
                        all_fields = display_all_fields(data[section_name])
                        if all_fields:
                            cols = st.columns(2)
                            col_idx = 0
                            for label, value in all_fields:
                                if isinstance(value, bool):
                                    display_value = "YES" if value else "NO"
                                    value_class = "true-value" if value else "false-value"
                                elif value == "" or value is None:
                                    display_value = "(empty)"
                                    value_class = "empty-field"
                                else:
                                    display_value = str(value)
                                    value_class = ""
                                with cols[col_idx % 2]:
                                    st.markdown(f"""
                                    <div class="field-container">
                                        <div class="field-label">{label}</div>
                                        <div class="field-value {value_class}">{display_value}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                col_idx += 1
            
            else:
                # For other pages, display normally
                for section_name, section_data in data.items():
                    if isinstance(section_data, dict):
                        st.markdown(f'<h3 class="section-header">{section_name.replace("_", " ").title()}</h3>', unsafe_allow_html=True)
                        
                        all_fields = display_all_fields(section_data)
                        if all_fields:
                            cols = st.columns(2)
                            col_idx = 0
                            for label, value in all_fields:
                                if isinstance(value, bool):
                                    display_value = "YES" if value else "NO"
                                    value_class = "true-value" if value else "false-value"
                                elif value == "" or value is None:
                                    display_value = "(empty)"
                                    value_class = "empty-field"
                                else:
                                    display_value = str(value)
                                    value_class = ""
                                
                                with cols[col_idx % 2]:
                                    st.markdown(f"""
                                    <div class="field-container">
                                        <div class="field-label">{label}</div>
                                        <div class="field-value {value_class}">{display_value}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                col_idx += 1
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_summary_metrics(results):
    """Display summary metrics at the top"""
    summary = {
        "Total Pages": len(results),
        "Principal Name": "Not Found",
        "CNIC": "Not Found",
        "Mobile": "Not Found"
    }
    
    for page_num, page_data in results.items():
        if "error" in page_data:
            continue
        
        if page_num == 1:
            if "principal_holder" in page_data:
                ph = page_data["principal_holder"]
                if ph.get("full_name"):
                    summary["Principal Name"] = ph["full_name"]
                if ph.get("cnic_nicop_passport"):
                    summary["CNIC"] = ph["cnic_nicop_passport"]
                if ph.get("mobile"):
                    summary["Mobile"] = ph["mobile"]
            
            if "type_of_account" in page_data:
                toa = page_data["type_of_account"]
                for acc_type, is_selected in toa.items():
                    if is_selected:
                        summary["Account Type"] = acc_type.replace('_', ' ').title()
                        break
    
    cols = st.columns(len(summary))
    for idx, (label, value) in enumerate(summary.items()):
        with cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=label, value=value)
            st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'extracted_results' not in st.session_state:
    st.session_state.extracted_results = None

# Upload box
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=["png", "jpg", "jpeg", "pdf"],
    help="Upload Al Meezan Investor Account Opening Form"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    file_type = uploaded_file.type
    
    if st.button("EXTRACT ALL DATA", type="primary", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("Please enter Google Gemini API Key in sidebar")
        else:
            if file_type == "application/pdf":
                st.info("Converting PDF to images...")
                images, num_pages = pdf_to_images(uploaded_file)
                if images and num_pages > 0:
                    st.success(f"Converted {min(num_pages, 5)} page(s)")
                    image_bytes_list = images[:5]
                else:
                    st.error("Failed to convert PDF")
                    st.stop()
            else:
                img_bytes = uploaded_file.read()
                image_bytes_list = [img_bytes]
            
            try:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                def update_progress(current, total):
                    progress_text.text(f"Processing page {current} of {total}...")
                    progress_bar.progress(current / total)
                
                with st.spinner("Extracting data..."):
                    results = extract_all_pages(image_bytes_list, "gemini-2.5-flash", update_progress)
                
                progress_text.empty()
                progress_bar.empty()
                
                st.session_state.extracted_results = results
                st.balloons()
                st.success("Extraction Complete!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Display extracted results
if st.session_state.extracted_results:
    st.markdown("---")
    st.markdown("## Extraction Summary")
    
    display_summary_metrics(st.session_state.extracted_results)
    
    st.markdown("---")
    st.markdown("## Extracted Form Data")
    st.caption("Showing ALL fields including empty values and unchecked checkboxes")
    st.markdown("---")
    
    for page_num in sorted(st.session_state.extracted_results.keys()):
        display_page_data(page_num, st.session_state.extracted_results[page_num])
    
    # Export section - CSV only
    st.markdown("---")
    st.markdown("## Export Data")
    
    csv_lines = ["Page,Category,Field,Value"]
    
    def add_to_csv(page_num, category, field, value):
        val_str = "YES" if isinstance(value, bool) and value else ("NO" if isinstance(value, bool) else str(value))
        csv_lines.append(f'"{page_num}","{category}","{field}","{val_str}"')
    
    for page_num, page_data in st.session_state.extracted_results.items():
        if "error" in page_data:
            continue
        
        for section, section_data in page_data.items():
            if isinstance(section_data, dict):
                all_fields = display_all_fields(section_data)
                for label, value in all_fields:
                    val_str = "YES" if isinstance(value, bool) and value else ("NO" if isinstance(value, bool) else str(value))
                    csv_lines.append(f'"{page_num}","{section}","{label}","{val_str}"')
    
    st.download_button(
        label="Download CSV",
        data="\n".join(csv_lines),
        file_name=f"al_meezan_form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.expander("View Raw JSON Data"):
        st.json(st.session_state.extracted_results)
    
    if st.button("Clear All Data", type="secondary", use_container_width=True):
        st.session_state.extracted_results = None
        st.rerun()

st.markdown("---")
st.markdown("**Note:** This tool extracts data page by page from Al Meezan Investment Account Opening Forms.")
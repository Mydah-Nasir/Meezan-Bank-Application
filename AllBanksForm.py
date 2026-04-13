import streamlit as st
from google import genai
import json
import os
import base64
from datetime import datetime
from PIL import Image
import io


st.set_page_config(page_title="Multi-Bank Form Parser", layout="wide")

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
st.markdown('<h1 class="main-header">Multi-Bank Form Extractor</h1>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
**AI-Powered Bank Form Processing** - Extract data from multiple Pakistani banks including Meezan, MCB, Allied, Alfalah, Askari, and more.
Upload an image or PDF and get structured data instantly.
""")

# Get Gemini API key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop()

# Bank selection
st.sidebar.image('OfficeFlow Ai-01-01.png', use_container_width='auto')
st.sidebar.markdown("### 🏦 Select Bank Form Type")
bank_option = st.sidebar.selectbox(
    "",
    [
        "Al Meezan Multi Page Form",
        "Meezan Bank", 
        "MCB Bank", 
        "Allied Bank", 
        "Alfalah Bank", 
        "Askari Bank", 
        "MCB Redemption C-1", 
        "MCB Early Redemption"
    ],
    help="Select the bank form type for optimized extraction"
)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Form Structure")
    st.caption("Each bank form has a specific structure optimized for extraction")
    
    if bank_option == "Al Meezan Multi Page Form":
        st.markdown("""
        - **Page 1:** Account Type, Principal Holder, Contact, Minor, Bank, Joint Holders
        - **Page 2:** KYC Details, PEP, Risk Profile, Next of Kin, Beneficiary, Declaration
        - **Page 3:** Checklist Documents
        - **Page 4:** FATCA Form
        - **Page 5:** CRS Tax Residency Form
        """)
    elif bank_option == "Meezan Bank":
        st.markdown("""
        - **Single Page Form**
        - Account Opening Form for Individual
        - Principal Account Holder Details
        - Contact & Bank Information
        - Joint Holders & Special Instructions
        """)
    elif bank_option == "MCB Bank":
        st.markdown("""
        - **Single Page Form**
        - Account Opening Form for Individuals
        - Personal & Contact Details
        - Bank Account & Guardian Details
        """)
    elif bank_option == "Allied Bank":
        st.markdown("""
        - **Single Page Form**
        - Personal Information
        - Employment Details
        - Contact Information
        """)
    elif bank_option == "Alfalah Bank":
        st.markdown("""
        - **Single Page Form**
        - Principal Applicant Details
        - Address & Contact Information
        - Tax & Zakat Status
        """)
    elif bank_option == "Askari Bank":
        st.markdown("""
        - **Single Page Form**
        - Account Holder Information
        - Bank Account Details
        - Joint Signatories & Nominees
        """)
    elif bank_option == "MCB Redemption C-1":
        st.markdown("""
        - **Single Page Form**
        - Redemption Request Form for Plans & Funds
        - Investor Registration Number
        - CNIC/NICOP/Passport
        - Fund & Redemption Details
        - CDS Account Information
        """)
    elif bank_option == "MCB Early Redemption":
        st.markdown("""
        - **Single Page Form**
        - Early Redemption Request Form
        - Fund Selection (Pakistan Pension / Alhamra Islamic)
        - Participant Details
        - Redemption Amount & Tax Information
        """)

# PDF conversion function
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

# ============================================
# PROMPTS FOR ALL BANK FORMS
# ============================================

# AL MEEZAN PROMPTS (Multi-page)
AL_MEEZAN_PAGE_1_PROMPT = """You are extracting data from PAGE 1 of Al Meezan Investor Account Opening Form.

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
  "joint_holders": [],
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

AL_MEEZAN_PAGE_2_PROMPT = """You are extracting data from PAGE 2 of Al Meezan Investor Account Opening Form.

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
    "principal_holder": {"no": false, "yes": false},
    "joint_holder_1": {"no": false, "yes": false},
    "joint_holder_2": {"no": false, "yes": false}
  },
  "risk_profile_age": {"above_60": false, "50_60": false, "40_50": false, "below_40": false},
  "risk_profile_tolerance": {
    "lower_risk_lower_returns": false,
    "medium_risk_medium_returns": false,
    "higher_risk_higher_returns": false
  },
  "risk_profile_savings": {"1000_25000": false, "25000_50000": false, "above_50000": false},
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
  "risk_profile_knowledge": {"limited_basic_average": false, "good_excellent": false},
  "risk_profile_horizon": {
    "less_than_6_months": false,
    "6_months_to_1_year": false,
    "1_to_3_years": false,
    "more_than_3_years": false
  },
  "risk_profile_result": {"total_score": null, "investor_portfolio": "", "recommended_fund": ""},
  "next_of_kin": {"name": "", "contact_number": "", "relation_with_customer": "", "address": ""},
  "beneficiary_details": {"ultimate_beneficiary_name": "", "relation_with_customer": "", "cnic_nicop_passport": ""},
  "declaration": {
    "principal_signature_present": false, 
    "joint_1_signature_present": false, 
    "joint_2_signature_present": false,
    "read_and_understood_guidelines": false
  },
  "office_use": {"sales_person_name": "", "dao_code": "", "manager_name": "", "reporting_date": "", "remarks": ""}
}"""

AL_MEEZAN_PAGE_3_PROMPT = """You are extracting data from PAGE 3 of Al Meezan Investor Account Opening Form.

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

AL_MEEZAN_PAGE_4_PROMPT = """You are extracting data from PAGE 4 (FATCA Form).

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

AL_MEEZAN_PAGE_5_PROMPT = """You are extracting data from PAGE 5 of Al Meezan Investor Account Opening Form - CRS Tax Residency Form.

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

# MEEZAN BANK PROMPT (Single Page)
MEEZAN_BANK_PROMPT = """You are extracting data from Meezan Bank Investor Account Opening Form for Individual.

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
  "joint_holders": [],
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

# MCB BANK PROMPT
MCB_BANK_PROMPT = """You are extracting data from MCB Funds Account Opening Form for Individuals.

Return ONLY valid JSON with this structure:

{
  "date": "",
  "purpose_new_account": false,
  "investor_registration_number": "",
  "principal_applicant": {
    "name": "",
    "father_spouse_name": "",
    "cnic_nicop_passport_bform": "",
    "mother_maiden_name": "",
    "gender_male": false,
    "gender_female": false,
    "gender_transgender": false,
    "date_of_birth": "",
    "zakat_deduction": false,
    "place_of_birth": "",
    "nationality": "",
    "share_percentage": "",
    "signature_present": false,
    "signature_as_per_cnic": false
  },
  "guardian_details": {
    "name": "",
    "relationship_with_minor": "",
    "cnic_nicop_passport": "",
    "nationality": ""
  },
  "contact_details": {
    "residential_address": "",
    "residential_city_district": "",
    "residential_postal_code": "",
    "residential_country": "",
    "office_business_address": "",
    "office_city_district": "",
    "office_postal_code": "",
    "office_country": "",
    "mailing_address_residential": false,
    "telephone_res": "",
    "telephone_off": "",
    "telephone_ext": "",
    "fax_no": "",
    "email_address": "",
    "mobile_no": ""
  },
  "statement_delivery_by_email": false,
  "bank_details": {
    "account_title": "",
    "complete_account_no": "",
    "bank_name": "",
    "branch_name_address": "",
    "city": "",
    "iban": ""
  },
  "joint_holders": [
    {
      "holder_number": 1,
      "name": "",
      "cnic_nicop_passport": "",
      "share_percentage": "",
      "gender_male": false,
      "gender_female": false,
      "gender_transgender": false,
      "signature_present": false,
      "signature_as_per_cnic": false,
      "declaration_read": false
    },
    {
      "holder_number": 2,
      "name": "",
      "cnic_nicop_passport": "",
      "share_percentage": "",
      "gender_male": false,
      "gender_female": false,
      "gender_transgender": false,
      "signature_present": false,
      "signature_as_per_cnic": false,
      "declaration_read": false
    },
    {
      "holder_number": 3,
      "name": "",
      "cnic_nicop_passport": "",
      "share_percentage": "",
      "gender_male": false,
      "gender_female": false,
      "gender_transgender": false,
      "signature_present": false,
      "signature_as_per_cnic": false,
      "declaration_read": false
    }
  ],
  "principal_applicant_signature_present": false,
  "thumb_impression_present": false,
  "declarations_read": false
}"""

# ALLIED BANK PROMPT
ALLIED_BANK_PROMPT = """You are extracting data from Allied Bank Account Opening Form for Individuals/Sole Proprietor.

Return ONLY valid JSON with this structure:

{
  "branch_name": "",
  "branch_code": "",
  "customer_number": "",
  "account_number": "",
  "date": "",
  "existing_customer": false,
  "existing_customer_info": "",
  "personal_information": {
    "full_name": "",
    "father_husband_name": "",
    "mother_maiden_name": "",
    "date_of_birth": "",
    "place_of_birth": "",
    "nationality": "",
    "country_of_residence": "",
    "residence_status_resident": false,
    "residence_status_non_resident": false,
    "cnic": "",
    "form_b_number": "",
    "passport_alien_card_poc": "",
    "nicop": "",
    "marital_status_single": false,
    "marital_status_married": false,
    "education_up_to_matric": false,
    "education_intermediate": false,
    "education_bachelors": false,
    "education_masters": false,
    "education_other": false,
    "profession_student": false,
    "profession_housewife": false,
    "profession_agriculturist": false,
    "profession_other": false,
    "profession_other_specify": "",
    "occupation_self_employed": false,
    "occupation_salaried": false,
    "occupation_other": false,
    "occupation_other_specify": ""
  },
  "employment_details": {
    "nature_of_business_retail": false,
    "nature_of_business_trading": false,
    "nature_of_business_services": false,
    "nature_of_business_manufacturing": false,
    "nature_of_business_self_employed_professional": false,
    "nature_of_business_other": false,
    "nature_of_business_other_specify": "",
    "business_employment_details_federal_government": false,
    "business_employment_details_provincial_government": false,
    "business_employment_details_semi_government": false,
    "business_employment_details_private_service": false,
    "business_employment_details_allied_bank_employee": false,
    "business_employment_details_other": false,
    "business_employment_details_other_specify": "",
    "name_of_business_organization": "",
    "designation": ""
  },
  "contact_information": {
    "business_office_address": "",
    "nearest_landmark": "",
    "post_code": "",
    "tel_no": "",
    "fax_no": "",
    "mob_no": "",
    "email": ""
  }
}"""

# ALFALAH BANK PROMPT
ALFALAH_BANK_PROMPT = """You are extracting data from Alfalah Investments Account Opening Form A-1.

Return ONLY valid JSON with this structure:

{
  "date": "",
  "investor_registration_no": "",
  "principal_applicant": {
    "name": "", "father_husband_name": "", "mother_maiden_name": "", "sole_proprietorship_name": "",
    "cnic_nicop_passport": "", "issuance_date": "", "expiry_date": "", "zakat_deduction": false,
    "tax_status_filer": false, "date_of_birth": "", "place_of_birth": "", "religion_muslim": false,
    "nationality": "", "national_tax_no": "", "marital_status_married": false, "gender_male": false
  },
  "address_details": {
    "current_mailing_address": "", "current_mailing_city": "", "current_mailing_province": "", "current_mailing_country": "",
    "permanent_address": "", "permanent_city": "", "permanent_province": "", "permanent_country": "",
    "business_address": ""
  },
  "contact_details": {"tel_no_res": "", "office_no": "", "mobile_no": "", "alternative_mobile_no": "", "whatsapp_no": "", "email": ""},
  "investor_signature_present": false
}"""

# ASKARI BANK PROMPT
ASKARI_BANK_PROMPT = """You are extracting data from Askari Investment Management Account Opening Form.

Return ONLY valid JSON with this structure:

{
  "date": {
    "day": "",
    "month": "",
    "year": ""
  },
  "principal_account_holder": {
    "full_name": "",
    "title": "",
    "father_husband_name": "",
    "type_of_institution_company": false,
    "type_of_institution_partnership": false,
    "type_of_institution_proprietorship": false,
    "type_of_institution_ngo": false,
    "type_of_institution_trust": false,
    "type_of_institution_others": false,
    "registration_incorporation_number": "",
    "ntn_number": "",
    "cnic_valid_passport": "",
    "mailing_address": "",
    "email": "",
    "landline_no_office": "",
    "landline_no_res": "",
    "mobile_no": "",
    "fax_no": "",
    "nationality": "",
    "marital_status_single": false,
    "marital_status_married": false,
    "country": "",
    "gender_male": false,
    "gender_female": false,
    "zakat_deduction_yes": false,
    "zakat_deduction_no": false,
    "date_of_birth": {
      "day": "",
      "month": "",
      "year": ""
    }
  },
  "bank_account_details": {
    "title_of_account": "",
    "bank_account_no": "",
    "bank_name": "",
    "branch": "",
    "bank_address": "",
    "bank_tel_no": ""
  },
  "joint_signatories": [
    {
      "name": "",
      "cnic_passport_no": ""
    }
  ],
  "nominee_information": [
    {
      "name": "",
      "relationship_with_principal": "",
      "cnic_no": "",
      "percentage": ""
    }
  ],
  "operating_instructions": {
    "singly": false,
    "principal_account_holder": false,
    "jointly_any_two": false,
    "jointly_all": false,
    "either_or_survivor": false,
    "others": false,
    "others_specify": ""
  },
  "other_instructions": {
    "payment_to_registered_address_yes": false,
    "payment_to_registered_address_no": false,
    "dividend_provide_cash": false,
    "dividend_reinvest": false,
    "payment_to_bank_yes": false,
    "payment_to_bank_no": false,
    "special_instructions": ""
  },
  "declaration": {
    "principal_authorised_signatory": "",
    "principal_authorised_signatory_name": "",
    "authorised_signatory_1": "",
    "authorised_signatory_1_name": "",
    "authorised_signatory_2": "",
    "authorised_signatory_2_name": "",
    "authorised_signatory_3": "",
    "authorised_signatory_3_name": ""
  },
  "distributor_info": {
    "distributor_name": "",
    "distributor_code": "",
    "facilitator_name": "",
    "facilitator_code": ""
  },
  "branch_info": {
    "branch_region": "",
    "mobile_no": "",
    "landline_no": "",
    "time": "",
    "date": ""
  },
  "remarks": ""
}"""

# MCB REDEMPTION C-1 PROMPT
MCB_REDEMPTION_C1_PROMPT = """You are extracting data from MCB Redemption Request Form For Plans And Funds - C-1.

Return ONLY valid JSON with this structure:

{
  "form_number": "",
  "date": "",
  "principal_applicant_details": {
    "title_of_account": "",
    "investor_registration_number": "",
    "cnic_nicop_passport": ""
  },
  "redemption_details": {
    "funds": [
      {
        "row_letter": "a",
        "name_of_fund_investment_plan": "",
        "type_of_units": "",
        "class_of_units": "",
        "number_of_units": "",
        "amount_in_figures": "",
        "amount_in_words": ""
      },
      {
        "row_letter": "b",
        "name_of_fund_investment_plan": "",
        "type_of_units": "",
        "class_of_units": "",
        "number_of_units": "",
        "amount_in_figures": "",
        "amount_in_words": ""
      },
      {
        "row_letter": "c",
        "name_of_fund_investment_plan": "",
        "type_of_units": "",
        "class_of_units": "",
        "number_of_units": "",
        "amount_in_figures": "",
        "amount_in_words": ""
      },
      {
        "row_letter": "d",
        "name_of_fund_investment_plan": "",
        "type_of_units": "",
        "class_of_units": "",
        "number_of_units": "",
        "amount_in_figures": "",
        "amount_in_words": ""
      }
    ],
    "certificates_issued": false,
    "certificate_number": "",
    "note_either_units_or_amount": "Please fill either No. of Units OR Amount. In case both are filled, amount will be considered for redemption"
  },
  "cds_account_details": {
    "participant_id_ias_id": "",
    "client_house_investor_account_no": ""
  },
  "declaration": {
    "understood_terms_conditions": false,
    "understood_capital_gain_tax": false,
    "understood_no_cancellation": false,
    "understood_cut_off_timings": false,
    "understood_identity_verification": false,
    "understood_bank_verification": false,
    "understood_redemption_based_on_holdings": false,
    "understood_pledge_certificate_cds_requirements": false
  },
  "investor_type": {
    "individual_investor": false,
    "institutional_investor": false,
    "company_stamp_present": false
  },
  "signatures": {
    "principal_applicant_signature_present": false,
    "left_hand_thumb_impression_present": false,
    "attestation_required": false,
    "attestation_of_branch_manager_present": false
  },
  "witnesses": [
    {
      "name": "",
      "cnic": "",
      "signature_present": false
    },
    {
      "name": "",
      "cnic": "",
      "signature_present": false
    }
  ],
  "authorized_signatories_joint_holders": [
    {
      "name": "",
      "signature_present": false
    }
  ],
  "investment_facilitator": {
    "distributor_facilitator_name": "",
    "code": "",
    "branch_name": "",
    "city": "",
    "distributor_stamp_present": false,
    "date": "",
    "time": ""
  },
  "registrar_details": {
    "form_received_by": "",
    "date_time_stamping": "",
    "form_verified_by": "",
    "data_input_by": ""
  },
  "proceeds_note": ""
}"""

# MCB EARLY REDEMPTION PROMPT
MCB_EARLY_REDEMPTION_PROMPT = """You are extracting data from MCB Early Redemption Request Form (FORM-VPS-07).

Return ONLY valid JSON with this structure:

{
  "form_metadata": {
    "institution": "MCB FUNDS",
    "form_title": "EARLY REDEMPTION REQUEST FORM",
    "form_code": "FORM-VPS-07",
    "version_date": ""
  },
  "date": "",
  "fund_selection": {
    "pakistan_pension_fund": false,
    "alhamra_islamic_pension_fund": false
  },
  "participant_details": {
    "participant_name": "",
    "registration_number": "",
    "ntn_no": "",
    "distinctive_account_number": ""
  },
  "early_redemption_information": {
    "unit_types": [
      {
        "unit_type": "Provident Fund",
        "selected": false,
        "amount_in_figures": "",
        "amount_in_words": ""
      },
      {
        "unit_type": "Non Provident Fund",
        "selected": false,
        "amount_in_figures": "",
        "amount_in_words": ""
      },
      {
        "unit_type": "Income Payment Plan",
        "selected": false,
        "amount_in_figures": "",
        "amount_in_words": ""
      }
    ],
    "instruction_note": "If no/incorrect unit Type is selected, unit as per statement of account will be redeemed"
  },
  "tax_details": {
    "applicability_note": "Mandatory Incase of Non Provident Fund Unit Type",
    "income_tax_returns_attached": false,
    "income_tax_returns_not_attached": false,
    "total_tax_paid_or_payable_three_years": "",
    "total_taxable_income_three_years": ""
  },
  "declaration": {
    "declarations_understood": [
      "terms_and_conditions",
      "withholding_tax_deduction",
      "no_cancellation_after_receipt",
      "cut_off_timings",
      "identity_verification_consent",
      "bank_mobile_verification_consent"
    ],
    "all_declarations_accepted": false
  },
  "investor_type": {
    "individual_investor": true,
    "institutional_investor": false
  },
  "signatures": {
    "principal_applicant_signature_present": false,
    "left_hand_thumb_impression_present": false,
    "attestation_required": false,
    "attestation_of_branch_manager_present": false
  },
  "witnesses": [
    {
      "name": "",
      "cnic": "",
      "signature_present": false
    },
    {
      "name": "",
      "cnic": "",
      "signature_present": false
    }
  ],
  "investment_facilitator": {
    "distributor_facilitator_name": "",
    "code": "",
    "branch_name": "",
    "city": "",
    "distributor_stamp_present": false,
    "date": "",
    "time": ""
  },
  "registrar_details": {
    "form_received_by": "",
    "date_time_stamping": "",
    "form_verified_by": "",
    "data_input_by": ""
  }
}"""

# Map banks to their prompts
BANK_PROMPTS = {
    "Al Meezan Multi Page Form": {
        "multi_page": True,
        "prompts": {
            1: AL_MEEZAN_PAGE_1_PROMPT,
            2: AL_MEEZAN_PAGE_2_PROMPT,
            3: AL_MEEZAN_PAGE_3_PROMPT,
            4: AL_MEEZAN_PAGE_4_PROMPT,
            5: AL_MEEZAN_PAGE_5_PROMPT
        }
    },
    "Meezan Bank": {"multi_page": False, "prompt": MEEZAN_BANK_PROMPT},
    "MCB Bank": {"multi_page": False, "prompt": MCB_BANK_PROMPT},
    "Allied Bank": {"multi_page": False, "prompt": ALLIED_BANK_PROMPT},
    "Alfalah Bank": {"multi_page": False, "prompt": ALFALAH_BANK_PROMPT},
    "Askari Bank": {"multi_page": False, "prompt": ASKARI_BANK_PROMPT},
    "MCB Redemption C-1": {"multi_page": False, "prompt": MCB_REDEMPTION_C1_PROMPT},
    "MCB Early Redemption": {"multi_page": False, "prompt": MCB_EARLY_REDEMPTION_PROMPT}
}

def extract_with_gemini(image_bytes, prompt, model_name="gemini-2.5-flash"):
    """Extract data using Gemini API"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    image = Image.open(io.BytesIO(image_bytes))
    
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, image]
    )
    
    response_text = response.text
    # Clean JSON response
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    return json.loads(response_text)

def extract_multi_page_form(image_bytes_list, prompts_dict, model_name, progress_callback=None):
    """Extract data from multi-page form"""
    all_results = {}
    for idx, img_bytes in enumerate(image_bytes_list):
        page_num = idx + 1
        if page_num > 5:
            break
        if progress_callback:
            progress_callback(page_num, min(len(image_bytes_list), 5))
        try:
            prompt = prompts_dict.get(page_num, prompts_dict[1])
            page_result = extract_with_gemini(img_bytes, prompt, model_name)
            all_results[page_num] = page_result
        except Exception as e:
            all_results[page_num] = {"error": str(e)}
    return all_results

def extract_single_page(image_bytes, prompt, model_name):
    """Extract data from single-page form"""
    try:
        return extract_with_gemini(image_bytes, prompt, model_name)
    except Exception as e:
        return {"error": str(e)}

# Display functions
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

def display_page_data(page_num, data):
    """Display ALL data for a page"""
    with st.container():
        st.markdown(f'<div class="form-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: #00A651;">Page {page_num}</h2>', unsafe_allow_html=True)
        
        if "error" in data:
            st.error(f"Error on this page: {data['error']}")
        else:
            for section_name, section_data in data.items():
                if isinstance(section_data, dict) and section_data:
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

def display_single_page_data(data, bank_name):
    """Display data for single-page form - handles both dict and list sections"""
    with st.container():
        st.markdown(f'<div class="form-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: #00A651;">{bank_name} - Extracted Data</h2>', unsafe_allow_html=True)
        
        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            for section_name, section_data in data.items():
                # Handle list sections (like joint_holders, funds, witnesses, etc.)
                if isinstance(section_data, list):
                    if section_data:
                        st.markdown(f'<h3 class="section-header">{section_name.replace("_", " ").title()}</h3>', unsafe_allow_html=True)
                        
                        # Display as columns for better readability
                        if len(section_data) <= 3:
                            cols = st.columns(len(section_data))
                            for idx, item in enumerate(section_data):
                                if isinstance(item, dict):
                                    with cols[idx]:
                                        st.markdown(f"**Item {idx + 1}**")
                                        all_fields = display_all_fields(item)
                                        for label, value in all_fields:
                                            if isinstance(value, bool):
                                                display_value = "YES" if value else "NO"
                                                value_class = "true-value" if value else "false-value"
                                            elif value == "" or value is None:
                                                display_value = "(not provided)"
                                                value_class = "empty-field"
                                            else:
                                                display_value = str(value)
                                                value_class = ""
                                            st.markdown(f"""
                                            <div class="field-container">
                                                <div class="field-label">{label.replace('_', ' ').title()}</div>
                                                <div class="field-value {value_class}">{display_value}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                        else:
                            # Display as expandable sections for large lists
                            for idx, item in enumerate(section_data):
                                if isinstance(item, dict):
                                    with st.expander(f"{section_name.replace('_', ' ').title()} - Item {idx + 1}"):
                                        all_fields = display_all_fields(item)
                                        cols = st.columns(2)
                                        col_idx = 0
                                        for label, value in all_fields:
                                            with cols[col_idx % 2]:
                                                if isinstance(value, bool):
                                                    display_value = "YES" if value else "NO"
                                                    value_class = "true-value" if value else "false-value"
                                                elif value == "" or value is None:
                                                    display_value = "(not provided)"
                                                    value_class = "empty-field"
                                                else:
                                                    display_value = str(value)
                                                    value_class = ""
                                                st.markdown(f"""
                                                <div class="field-container">
                                                    <div class="field-label">{label.replace('_', ' ').title()}</div>
                                                    <div class="field-value {value_class}">{display_value}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            col_idx += 1
                    else:
                        st.markdown(f'<h3 class="section-header">{section_name.replace("_", " ").title()}</h3>', unsafe_allow_html=True)
                        st.markdown('<div class="field-container"><div class="field-value empty-field">No data available</div></div>', unsafe_allow_html=True)
                
                # Handle dictionary sections
                elif isinstance(section_data, dict) and section_data:
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
                                display_value = "(not provided)"
                                value_class = "empty-field"
                            else:
                                display_value = str(value)
                                value_class = ""
                            
                            with cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="field-container">
                                    <div class="field-label">{label.replace('_', ' ').title()}</div>
                                    <div class="field-value {value_class}">{display_value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            col_idx += 1
                
                # Handle empty dict sections
                elif isinstance(section_data, dict) and not section_data:
                    st.markdown(f'<h3 class="section-header">{section_name.replace("_", " ").title()}</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="field-container"><div class="field-value empty-field">No data available</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_summary_metrics(results, bank_name):
    """Display summary metrics at the top"""
    summary = {
        "Bank": bank_name,
        "Total Pages": len(results) if isinstance(results, dict) else 1,
        "Status": "Extracted"
    }
    
    cols = st.columns(len(summary))
    for idx, (label, value) in enumerate(summary.items()):
        with cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=label, value=value)
            st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'extracted_results' not in st.session_state:
    st.session_state.extracted_results = None
if 'current_bank' not in st.session_state:
    st.session_state.current_bank = None

# Upload box
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=["png", "jpg", "jpeg", "pdf"],
    help=f"Upload {bank_option} form"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    file_type = uploaded_file.type
    bank_config = BANK_PROMPTS[bank_option]
    
    if st.button("EXTRACT ALL DATA", type="primary", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("API Key not configured")
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
                    if bank_config["multi_page"]:
                        results = extract_multi_page_form(
                            image_bytes_list, 
                            bank_config["prompts"], 
                            "gemini-2.5-flash",
                            update_progress
                        )
                    else:
                        results = extract_single_page(
                            image_bytes_list[0], 
                            bank_config["prompt"], 
                            "gemini-2.5-flash"
                        )
                
                progress_text.empty()
                progress_bar.empty()
                
                st.session_state.extracted_results = results
                st.session_state.current_bank = bank_option
                st.success("Extraction Complete!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Display extracted results
if st.session_state.extracted_results:
    st.markdown("---")
    st.markdown("## Extraction Summary")
    
    display_summary_metrics(st.session_state.extracted_results, st.session_state.current_bank)
    
    st.markdown("---")
    st.markdown("## Extracted Form Data")
    st.caption("Showing ALL fields including empty values and unchecked checkboxes")
    st.markdown("---")
    
    bank_config = BANK_PROMPTS[st.session_state.current_bank]
    
    if bank_config["multi_page"]:
        for page_num in sorted(st.session_state.extracted_results.keys()):
            display_page_data(page_num, st.session_state.extracted_results[page_num])
    else:
        display_single_page_data(st.session_state.extracted_results, st.session_state.current_bank)
    
    # Export section
    st.markdown("---")
    st.markdown("## Export Data")
    
    csv_lines = ["Page,Category,Field,Value"]
    
    def add_to_csv(page_num, category, field, value):
        val_str = "YES" if isinstance(value, bool) and value else ("NO" if isinstance(value, bool) else str(value))
        csv_lines.append(f'"{page_num}","{category}","{field}","{val_str}"')
    
    if bank_config["multi_page"]:
        for page_num, page_data in st.session_state.extracted_results.items():
            if "error" in page_data:
                continue
            for section, section_data in page_data.items():
                if isinstance(section_data, dict):
                    all_fields = display_all_fields(section_data)
                    for label, value in all_fields:
                        add_to_csv(page_num, section, label, value)
    else:
        for section, section_data in st.session_state.extracted_results.items():
            if isinstance(section_data, dict):
                all_fields = display_all_fields(section_data)
                for label, value in all_fields:
                    add_to_csv(1, section, label, value)
    
    st.download_button(
        label="📥 Download CSV",
        data="\n".join(csv_lines),
        file_name=f"{st.session_state.current_bank.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.expander("🔍 View Raw JSON Data"):
        st.json(st.session_state.extracted_results)
    
    if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        st.session_state.extracted_results = None
        st.session_state.current_bank = None
        st.rerun()

st.markdown("---")
st.markdown("**Note:** This tool extracts data from multiple bank forms. Results may vary based on form quality.")
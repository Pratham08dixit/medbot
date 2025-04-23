
import streamlit as st
st.set_page_config(page_title="MedBot", page_icon="💊")
import re
from datetime import datetime
from dotenv import load_dotenv
import os 
import sqlite3
import pandas as pd
import numpy as np
import google.generativeai as genai

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# ============================ Load Env & API ============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY in .env")
genai.configure(api_key=GOOGLE_API_KEY)

# ============================ File Paths ============================
FORECAST_DB = "forecasts.db"
INV_CSV = "stock_medicine_inventory.csv"
USAGE_CSV = "sample_drug_usage_with_dates.csv"
CURRENT_INV_CSV    = "current_inventory.csv"  

usage_df = pd.read_csv(USAGE_CSV, parse_dates=["date"])
inv_df = pd.read_csv(INV_CSV, parse_dates=True)
current_inv_df  = pd.read_csv(CURRENT_INV_CSV)

DRUG_COL = next((c for c in inv_df.columns if "drug" in c.lower()), None)
MFG_COL = next((c for c in inv_df.columns if "manufactur" in c.lower() or "mfg" in c.lower()), None)
EXP_COL = next((c for c in inv_df.columns if "expiry" in c.lower() or "exp" in c.lower()), None)
BOX_COL = next((c for c in inv_df.columns if "box" in c.lower()), None)
CURR_INV_COL = next((c for c in current_inv_df.columns if "current_inventory" in c.lower() or "current stock" in c.lower()), None)

_drug_list = sorted(
    set(inv_df[DRUG_COL].str.lower()) |
    set(c.lower() for c in usage_df.columns if c.lower() != "date"),
    key=lambda s: -len(s)
)

def extract_drug(text: str) -> str:
    txt = text.lower()
    for d in _drug_list:
        if re.search(rf"\b{re.escape(d)}\b", txt):
            return d
    return None

def is_greeting(text):
    return any(word in text.lower() for word in ["hi", "hello", "hey", "greetings", "good morning", "good evening"])

# ============================ LLM Wrapper ============================
class GoogleGenerativeLLM(LLM):
    model_name: str = "gemini-2.0-flash"
    max_output_tokens: int = 750

    @property
    def _llm_type(self): return "google_generative"
    @property
    def _identifying_params(self): return {"model_name": self.model_name, "max_output_tokens": self.max_output_tokens}

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content(prompt)
        if not getattr(resp, "text", None):
            raise ValueError("No response from Google Generative AI.")
        return resp.text

def direct_llm_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    r = model.generate_content(prompt)
    return getattr(r, "text", str(r))

# ============================ Document Loaders ============================
def load_inventory_documents_sql():
    conn = sqlite3.connect(FORECAST_DB)
    df = pd.read_sql_query("SELECT * FROM forecasts", conn)
    conn.close()
    return [Document(page_content=(
        f"Drug: {r['drug']}\n"
        f"Current Inventory: {r['current_inventory']}\n"
        f"Forecast: {r['forecast_day1']}, {r['forecast_day2']}, {r['forecast_day3']}\n"
        f"Total Forecast: {r['total_forecast']}\n"
        f"Reorder Qty: {r['reorder_quantity']}\n"
    )) for _, r in df.iterrows()]

def load_inventory_documents_csv():
    return [Document(page_content=(
        f"Drug: {r[DRUG_COL]}\n"
        f"Manufacturing Date: {pd.to_datetime(r[MFG_COL]).strftime('%Y-%m-%d')}\n"
        f"Expiry Date: {pd.to_datetime(r[EXP_COL]).strftime('%Y-%m-%d')}\n"
    )) for _, r in inv_df.iterrows()]

def load_usage_documents_with_dates():
    docs = []
    for c in usage_df.columns:
        if c.lower() == "date": continue
        text = "".join(f"{r['date'].strftime('%Y-%m-%d')}: {r[c]}\n" for _, r in usage_df.iterrows())
        docs.append(Document(page_content=f"Usage for {c}:\n{text}"))
    return docs

def load_current_inventory_documents():
    """
    Reads CURRENT_INV_CSV (with columns 'Drug' & 'Current_Inventory')
    and returns a list of Document objects where each page_content is:

      Drug: <drug_name>
      Current Inventory: <inventory_value>
    """
    # assume CURRENT_INV_CSV is already defined and loaded as `current_inv_df`
    docs = []
    for _, row in current_inv_df.iterrows():
        drug      = row['Drug']
        inventory = row['Current_Inventory']
        content   = (
            f"Drug: {drug}\n"
            f"Current Inventory: {inventory}"
        )
        docs.append(Document(page_content=content))
    return docs


def build_vector_store():
    docs = []
    docs += load_inventory_documents_sql()
    docs += load_inventory_documents_csv()
    docs += load_usage_documents_with_dates()
    docs += load_current_inventory_documents()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

# ============================ Handlers ============================
def check_reorder_requirement(drug):
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT reorder_quantity FROM forecasts WHERE LOWER(drug)=?", (drug,))
    row = cur.fetchone(); conn.close()
    if not row: return f"No reorder data for '{drug}'."
    qty = row[0]
    return f"Reorder needed: {'Yes' if qty > 0 else 'No'}. Quantity: {int(qty)} units."

def simple_reorder_check(drug):
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT reorder_quantity FROM forecasts WHERE LOWER(drug)=?", (drug,))
    row = cur.fetchone(); conn.close()
    if not row: return f"No reorder info for '{drug}'."
    reorder_qty = row[0]
    return f"✅ Yes, you need to reorder **{drug.title()}**. Recommended: {reorder_qty} units." if reorder_qty > 0 else f"🟢 No need to reorder **{drug.title()}** right now."

def check_forecast(drug):
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT forecast_day1,forecast_day2,forecast_day3,total_forecast FROM forecasts WHERE LOWER(drug)=?", (drug,))
    row = cur.fetchone(); conn.close()
    if not row: return f"No forecast data for '{drug}'."
    d1, d2, d3, total = row
    return f"Forecast for {drug}:\n • Day 1: {d1}\n • Day 2: {d2}\n • Day 3: {d3}\n • Total: {total}"

def list_forecast_less(threshold: int) -> str:
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT drug FROM forecasts WHERE total_forecast < ?", (threshold,))
    rows = [r[0] for r in cur.fetchall()]; conn.close()
    if not rows: return f"No medicines have total forecast < {threshold}."
    names = ', '.join(r.title() for r in rows)
    return f"Medicines with total forecast < {threshold}: {names}."

def list_forecast_greater(threshold: int) -> str:
    """
    Lists all medicines whose total forecast is strictly greater than the given threshold.
    """
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT drug FROM forecasts WHERE total_forecast > ?", (threshold,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    if not rows:
        return f"No medicines have total forecast > {threshold}."
    names = ", ".join(drug.title() for drug in rows)
    return f"Medicines with total forecast > {threshold}: {names}."

def list_reorder_less(threshold: int) -> str:
    """
    Lists all medicines whose reorder quantity is strictly less than the given threshold.
    """
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT drug FROM forecasts WHERE reorder_quantity < ?", (threshold,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    if not rows:
        return f"No medicines have reorder quantity < {threshold}."
    names = ", ".join(drug.title() for drug in rows)
    return f"Medicines with reorder quantity < {threshold}: {names}."

def list_reorder_greater(threshold: int) -> str:
    """
    Lists all medicines whose reorder quantity is strictly greater than the given threshold.
    """
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT drug FROM forecasts WHERE reorder_quantity > ?", (threshold,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    if not rows:
        return f"No medicines have reorder quantity > {threshold}."
    names = ", ".join(drug.title() for drug in rows)
    return f"Medicines with reorder quantity > {threshold}: {names}."

def check_usage_by_date(drug, date_str):
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y").date()
    except ValueError:
        return "Date must be in dd/mm/yyyy format."
    mask = usage_df["date"].dt.date == dt
    row = usage_df.loc[mask]
    if row.empty:
        return f"No usage data for '{drug}' on {date_str}."
    usage_col = next((c for c in usage_df.columns if c.strip().lower() == drug.strip().lower()), None)
    if not usage_col:
        return f"No usage data for '{drug}'."
    val = row.iloc[0][usage_col]
    return f"Usage for {drug} on {date_str}: {val}"

def check_usage_by_day(drug, day):
    if day < 1 or day > len(usage_df):
        return f"Day {day} out of range (1–{len(usage_df)})."
    row = usage_df.iloc[day - 1]
    date_str = row["date"].strftime("%d/%m/%Y")
    usage_col = next((c for c in usage_df.columns if c.lower() == drug), None)
    if not usage_col:
        return f"No usage data for '{drug}'."
    val = row[usage_col]
    return f"Day {day} ({date_str}) usage for {drug}: {val}"

def check_mfg_date(drug):
    sub = inv_df[inv_df[DRUG_COL].str.lower() == drug]
    if sub.empty:
        return f"No manufacturing data for '{drug}'."
    return f"Manufacturing date for {drug}: {pd.to_datetime(sub.iloc[0][MFG_COL]).strftime('%d/%m/%Y')}"

def check_expiry_date(drug):
    sub = inv_df[inv_df[DRUG_COL].str.lower() == drug]
    if sub.empty:
        return f"No expiry data for '{drug}'."
    return f"Expiry date for {drug}: {pd.to_datetime(sub.iloc[0][EXP_COL]).strftime('%d/%m/%Y')}"

def list_expiry_before(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str, format="%d/%m/%Y")
    except ValueError:
        return "Date must be in dd/mm/YYYY format."
    series = pd.to_datetime(inv_df[EXP_COL], errors='coerce')
    subs   = inv_df[series < dt]
    if subs.empty:
        return f"No medicines expiring before {date_str}."
    meds = subs[DRUG_COL].dropna().unique()
    meds = sorted(m.title() for m in meds)
    return f"Medicines expiring before {date_str}: {', '.join(meds)}."

def list_mfg_before(date_str: str) -> str:
    """
    Lists all medicines whose manufacturing date is strictly before the given date.

    Args:
        date_str (str): Date in dd/mm/YYYY format.

    Returns:
        str: A formatted string listing matching drugs, or a message if none found.
    """
    try:
        dt = pd.to_datetime(date_str, format="%d/%m/%Y")
    except ValueError:
        return "Date must be in dd/mm/YYYY format."
    series = pd.to_datetime(inv_df[MFG_COL], errors='coerce')
    subs = inv_df[series < dt]
    if subs.empty:
        return f"No medicines manufactured before {date_str}."
    meds = subs[DRUG_COL].dropna().unique()
    meds = sorted(m.title() for m in meds)
    return f"Medicines manufactured before {date_str}: {', '.join(meds)}."

def check_no_of_boxes(drug):
    sub = inv_df[inv_df[DRUG_COL].str.lower() == drug]
    if sub.empty:
        return f"Sorry, I couldn't find any box data for '{drug}'."
    if BOX_COL is None:
        return "Box quantity column not found in inventory file."
    qty = sub.iloc[0][BOX_COL]
    return f"📦 Number of boxes of {drug.title()}: {qty}"

# Current inventory
def check_current_inventory(drug):
    sub = current_inv_df[current_inv_df['Drug'].str.lower()==drug]
    if sub.empty: return f"No current inventory info for '{drug}'."
    if CURR_INV_COL is None: return "Current_Inventory column missing."
    return f"🔢 Current inventory for {drug.title()}: {sub.iloc[0][CURR_INV_COL]} units."

def is_general_medical_query(q):
    ql = q.lower()
    kws = [
        "symptom", "treat", "treatment", "diet", "cure", "prevent", "side effect", "allergy", "suggest",
        "prevention", "medicine", "medication", "drug", "salt formula", "recommend",
        "formula", "use", "dosage", "salt", "composition", "chemical", "consult", "cure", "health", "disease", "accident", "injury", "weight", "gain", "loss", "bones"
    ]
    return any(kw in ql for kw in kws)

# ============================ LangChain Setup ============================
@st.cache_resource
def setup_chain():
    memory = ConversationBufferMemory(memory_key="chat_history")
    vs = build_vector_store()
    llm = GoogleGenerativeLLM()
    prompt = PromptTemplate.from_template(
        """
You are a medical domain expert and helpful assistant with personality.
Answer only queries about drug usage, inventory, reorder, forecasts, packaging, etc.
If someone greets, greet back politely with emoji.
Reject unrelated queries.

Context:
{context}

Question: {question}

Answer (structured, no * or $):
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k":4}),
        return_source_documents=False,
        memory=memory,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

qa_chain = setup_chain()

# ============================ Streamlit Chat Interface ============================
#st.set_page_config(page_title="MedBot", page_icon="💊")
st.title("💊 MedBot - AI Medical Assistant")
col1, col2 = st.columns([0.8, 0.2])
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything...")

if user_input:
    ql = user_input.lower()
    drug = extract_drug(user_input)
    date_m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", user_input)
    day_m = re.search(r"(\d+)(?:st|nd|rd|th)?\s*day\b", ql)
    forecast_less_m = re.search(r"(?:forecast(ed)?|total forecast).*less than (\d+)", ql)
    date_q = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", user_input)

    if is_greeting(ql):
        answer = "Hey there! 👋 Ready to assist with your pharmacy data, How can i help you ?"
    elif drug and any(k in ql for k in ["current inventory","inventory", "current stock"]):
        answer = check_current_inventory(drug)
    elif date_m and any(k in ql for k in ["expir", "expiry", "expiring", "exp"]):
        answer = list_expiry_before(date_m.group(1))
    elif forecast_less_m:
        answer = list_forecast_less(int(forecast_less_m.group(2)))
    elif re.search(r"(?:forecast(?:ed)?|total forecast).*(?:greater than|more than)\s+(\d+)", ql):
        n = int(re.search(r"(?:greater than|more than)\s+(\d+)", ql).group(1))
        answer = list_forecast_greater(n)
    elif re.search(r"reorder.*less than (\d+)", ql):
        n = int(re.search(r"less than (\d+)", ql).group(1))
        answer = list_reorder_less(n)
    elif re.search(r"reorder.*greater than (\d+)", ql):
        n = int(re.search(r"greater than (\d+)", ql).group(1))
        answer = list_reorder_greater(n)
    elif date_q and any(k in ql for k in ["manufactur", "mfg"]) and "before" in ql:
        answer = list_mfg_before(date_q.group(1))
    elif any(k in ql for k in ["reorder","restock","order again"] ) and drug:
        if any(phrase in ql for phrase in ["do i need to reorder","should i reorder","need to order"]):
            answer = simple_reorder_check(drug)
        else:
            answer = check_reorder_requirement(drug)
    elif any(k in ql for k in ["forecast","predict"] ) and drug:
        answer = check_forecast(drug)
    elif date_q and drug and "usage" in ql:
        answer = check_usage_by_date(drug, date_q.group(1))
    elif day_m and drug:
        answer = check_usage_by_day(drug, int(day_m.group(1)))
    elif any(k in ql for k in ["manufactur","production date", "mfg"] ) and drug:
        answer = check_mfg_date(drug)
    elif any(k in ql for k in ["expir","expiration date", "exp"] ) and drug:
        answer = check_expiry_date(drug)
    elif any(k in ql for k in ["boxes","box count"] ) and drug:
        answer = check_no_of_boxes(drug)
    elif is_general_medical_query(user_input):
        raw = direct_llm_response(
            "You are a specialist medical assistant. Provide a clear, strucuted answer "
            "Without special characters (*, $, etc.).\n "
            f"Question: {user_input}\nAnswer:\n"
        )
        answer = re.sub(r"[\*\$]","", raw).strip()
    else:
        answer = "🤔 I didn't catch that. Could you rephrase your medical or inventory-related question?"

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# Display chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

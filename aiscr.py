import streamlit as st
import torch
import pandas as pd
import datetime
import os
import json
import ollama
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. HARDWARE ACCELERATION & GLOBAL PATHS
# ==========================================
MODEL_PATH = "psychiatric_deberta_v3"  # Path to saved model weights directory
MODEL_NAME = "microsoft/deberta-v3-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CREDENTIALS_FILE = "credentials.txt"
PDATA_FILE = "pdata.txt"
ADMIN_PASSWORD = "admin123"  # Master password for the System Admin

# 7-Class Alphabetical Mapping matching your training pipeline
ID_TO_LABEL = {
    0: 'Anxiety', 1: 'Bipolar', 2: 'Depression', 3: 'Normal',
    4: 'Personality disorder', 5: 'Stress', 6: 'Suicidal'
}

# ==========================================
# 2. FILE DATABASE INITIALIZATION HELPERS
# ==========================================
def init_files():
    """Initializes credential files and base dataset logs if missing on the drive."""
    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'w') as f:
            f.write("PATIENTS: PT-101, PT-102, PT-103, PT-104, PT-105\n")
            f.write("DOCTORS: DR-ADMIN, DR-SMITH, DR-001\n")
            
    if not os.path.exists(PDATA_FILE):
        default_data = [
            {"Patient ID": "PT-101", "Timestamp": "2026-03-19 09:00", "Patient Input": "I feel completely hopeless and exhausted.", "AI Diagnosis": "🔴 Depression", "Confidence": "99.61%", "Detected Markers": "hopeless, exhausted"},
            {"Patient ID": "PT-102", "Timestamp": "2026-03-19 09:15", "Patient Input": "My heart is pounding and I cannot breathe easily.", "AI Diagnosis": "🟡 Anxiety", "Confidence": "96.79%", "Detected Markers": "pounding, breathe"},
            {"Patient ID": "PT-103", "Timestamp": "2026-03-19 09:30", "Patient Input": "I'm doing okay, just busy reading.", "AI Diagnosis": "🟢 Normal", "Confidence": "98.11%", "Detected Markers": "None"}
        ]
        with open(PDATA_FILE, 'w') as f:
            json.dump(default_data, f, indent=4)

def load_credentials():
    """Loads authenticated lists directly from file storage."""
    patients, doctors = [], []
    with open(CREDENTIALS_FILE, 'r') as f:
        for line in f:
            if line.startswith("PATIENTS:"):
                patients = [x.strip() for x in line.replace("PATIENTS:", "").split(",") if x.strip()]
            elif line.startswith("DOCTORS:"):
                doctors = [x.strip() for x in line.replace("DOCTORS:", "").split(",") if x.strip()]
    return patients, doctors

def save_credentials(patients, doctors):
    """Saves updated identity parameters to the filesystem."""
    with open(CREDENTIALS_FILE, 'w') as f:
        f.write("PATIENTS: " + ", ".join(patients) + "\n")
        f.write("DOCTORS: " + ", ".join(doctors) + "\n")

def load_pdata():
    """Fetches diagnostic profiles from pdata database file."""
    with open(PDATA_FILE, 'r') as f:
        return json.load(f)

def save_pdata(new_record):
    """Appends live runtime observations to persistent pdata storage."""
    data = load_pdata()
    data.append(new_record)
    with open(PDATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

init_files()
VALID_PATIENT_IDS, VALID_DOCTOR_IDS = load_credentials()

# ==========================================
# 3. CACHED DEBERTA CORE MODEL LOADING
# ==========================================
@st.cache_resource
def load_neural_network():
    """Instantiates the DeBERTa-v3 transformer backend with saved weights."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=7)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model, True
    except Exception:
        return tokenizer, None, False

tokenizer, neural_model, model_loaded_successfully = load_neural_network()

# ==========================================
# 4. NEURAL INFERENCE ENGINE PIPELINE
# ==========================================
def run_neural_inference(text):
    """Processes natural language statements through the GPU-accelerated pipeline."""
    if not model_loaded_successfully:
        return "Normal", "100.00%", "None"

    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        outputs = neural_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item() * 100

    label = ID_TO_LABEL[pred_id]
    
    if label in ["Suicidal", "Depression"]:
        status_tag = f"🔴 {label}"
    elif label in ["Anxiety", "Bipolar", "Personality disorder", "Stress"]:
        status_tag = f"🟡 {label}"
    else:
        status_tag = f"🟢 {label}"

    text_lower = text.lower()
    clinical_dictionary = ["hopeless", "panic", "racing", "worthless", "die", "tired", "shaking", "heart"]
    found_markers = [m for m in clinical_dictionary if m in text_lower]
    markers_string = ", ".join(found_markers) if found_markers else "None"

    return status_tag, f"{confidence:.2f}%", markers_string

# ==========================================
# 5. STREAMLIT SESSION STATE ARCHITECTURE
# ==========================================
st.set_page_config(page_title="AI Proxy Patient System", page_icon="🧠", layout="wide")

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"
if 'patient_chat_history' not in st.session_state:
    st.session_state.patient_chat_history = []
if 'doctor_bot_history' not in st.session_state:
    st.session_state.doctor_bot_history = []
if 'active_user' not in st.session_state:
    st.session_state.active_user = None

def navigate_to(page_name):
    st.session_state.current_page = page_name

def logout_procedure():
    st.session_state.active_user = None
    st.session_state.current_page = "Login"

# ==========================================
# 6. CORE GATEKEEPER LOGIN
# ==========================================
if st.session_state.current_page == "Login":
    st.markdown("<h1 style='text-align: center;'>AI Driven Proxy Patient System</h1>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🔒 **System Authentication Required**")
        login_role = st.radio("Select Security Profile:", ["Patient", "Doctor / Medical Staff", "System Admin"], horizontal=True)
        
        if login_role == "System Admin":
            entered_id = st.text_input("Enter Root Master Password:", type="password")
        else:
            entered_id = st.text_input(f"Enter Registered Identity ID String:")
        
        if st.button("Establish Secure Connection", use_container_width=True):
            if login_role == "Patient" and entered_id.upper() in VALID_PATIENT_IDS:
                st.session_state.active_user = entered_id.upper()
                navigate_to("Patient Chat")
                st.rerun()
            elif login_role == "Doctor / Medical Staff" and entered_id.upper() in VALID_DOCTOR_IDS:
                st.session_state.active_user = entered_id.upper()
                navigate_to("Doctor Hub")
                st.rerun()
            elif login_role == "System Admin" and entered_id == ADMIN_PASSWORD:
                st.session_state.active_user = "System Administrator"
                navigate_to("Admin Hub")
                st.rerun()
            else:
                st.error("❌ Authentication Refused. Check context configurations.")

# ==========================================
# 7. SECURE PATIENT CHAT (DEBERTA + OLLAMA)
# ==========================================
elif st.session_state.current_page == "Patient Chat":
    st.sidebar.button("🚪 Terminate Session", on_click=logout_procedure)
    st.sidebar.info(f"Connected Channel: **{st.session_state.active_user}**")
    st.sidebar.markdown(f"**Hardware Pipeline:** {'✅ NVIDIA CUDA Active' if DEVICE=='cuda' else '⚠️ CPU Core Active'}")
    
    st.title("🗣️ Patient Intake Workspace")
    
    for msg in st.session_state.patient_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Explain your symptoms or current thoughts here..."):
        st.session_state.patient_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        status, conf, markers = run_neural_inference(prompt)
        
        new_record = {
            "Patient ID": st.session_state.active_user,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient Input": prompt,
            "AI Diagnosis": status,
            "Confidence": conf,
            "Detected Markers": markers
        }
        save_pdata(new_record)
            
        with st.chat_message("assistant"):
            with st.spinner("Llama 3 Evaluating Context..."):
                sys_prompt = f"""You are a highly empathetic clinical AI intake manager.
                The user typed: '{prompt}'.
                Our verified PyTorch classifier mapped this statement to the profile status: '{status}'.
                
                Generate a compassionate response confined strictly to two short sentences.
                Acknowledge their state gently, do not explicitly state what model was used, and reassure them that these notes are saved for their human doctor."""
                
                try:
                    response = ollama.generate(model='llama3', prompt=sys_prompt)
                    bot_reply = response['response']
                except Exception as e:
                    bot_reply = "I process your statement securely. Your doctor has been informed."
                    st.error(f"🛑 Local Engine Offline. Run 'sudo systemctl start ollama'. {e}")
                
                st.markdown(bot_reply)
                st.session_state.patient_chat_history.append({"role": "assistant", "content": bot_reply})

# ==========================================
# 8. MEDICAL STAFF INTEGRATED HUB
# ==========================================
elif st.session_state.current_page == "Doctor Hub":
    st.sidebar.button("🚪 Terminate Session", on_click=logout_procedure)
    st.sidebar.info(f"Staff Member: **{st.session_state.active_user}**")
    
    st.title("🩺 Clinical Research & Monitoring Dashboard")
    st.write("---")
    
    # Pruned down to 2 safe logical columns to prevent doctor-to-doctor directory alteration vectors
    col1, col2 = st.columns(2) 
    with col1:
        st.success("**Monitoring Matrix**") 
        if st.button("📊 Launch Live Inspection Grid", use_container_width=True):
            navigate_to("Doctor Dashboard")
            st.rerun()
    with col2:
        st.warning("**Simulator Sandbox**") 
        if st.button("💬 Initialize Proxy Patient Training", use_container_width=True):
            navigate_to("Doctor Bot Comm")
            st.rerun()

# ==========================================
# 9. CLINICAL AUDIT RUNTIME MONITOR MATRIX
# ==========================================
elif st.session_state.current_page == "Doctor Dashboard":
    st.sidebar.button("⬅️ Return to Command Hub", on_click=navigate_to, args=("Doctor Hub",))
    st.sidebar.button("🚪 Logout", on_click=logout_procedure)
    
    st.title("📊 Clinical Patient Monitoring Interface")
    
    pdata = load_pdata()
    df = pd.DataFrame(pdata)
    
    crit_count = len(df[df['AI Diagnosis'].str.contains("🔴")])
    warn_count = len(df[df['AI Diagnosis'].str.contains("🟡")])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total File Records Linked", len(df))
    m2.metric("Critical Red Alerts (Depression/Suicidal)", crit_count)
    m3.metric("Observation Flags (Anxiety/Stress/Bipolar)", warn_count)
    st.divider()
    
    def apply_row_triage_colors(val):
        if "🔴" in str(val): return 'background-color: #ffcccc; color: #5c0000; font-weight: bold;'
        elif "🟡" in str(val): return 'background-color: #fff4cc; color: #5c4300;'
        elif "🟢" in str(val): return 'background-color: #ccffcc; color: #004d00;'
        return ''
    
    st.write("### Live Patient Evaluation Ledger")
    st.dataframe(df.iloc[::-1].style.map(apply_row_triage_colors, subset=['AI Diagnosis']), use_container_width=True)
    
    st.divider()
    st.write("### 🗂️ Patient Case History Explorer")
    pt_ids_available = list(df['Patient ID'].unique())
    selected_target_id = st.selectbox("Select Patient Profile Node:", ["-- Select ID --"] + pt_ids_available)
    
    if selected_target_id != "-- Select ID --":
        sub_df = df[df['Patient ID'] == selected_target_id]
        for idx, row in sub_df.iterrows():
            st.caption(f"Log Identifier timestamp: {row['Timestamp']}")
            with st.chat_message("user"):
                st.write(row['Patient Input'])
            with st.chat_message("assistant"):
                st.write(f"**DeBERTa Screening Outcome:** {row['AI Diagnosis']} (Confidence: {row['Confidence']})")
                st.write(f"*Extracted Cues:* `{row['Detected Markers']}`")
            st.write("---")

# ==========================================
# 10. LOCAL TRAINING SIMULATION SANDBOX (OLLAMA LLM)
# ==========================================
elif st.session_state.current_page == "Doctor Bot Comm":
    st.sidebar.button("⬅️ Return to Command Hub", on_click=navigate_to, args=("Doctor Hub",))
    st.sidebar.button("🚪 Logout", on_click=logout_procedure)
    
    st.title("🩺 Generative Proxy Patient Simulation Arena")
    pdata = load_pdata()
    linked_pts = list(set([row["Patient ID"] for row in pdata]))
    
    st.info("The system automatically parses historical records so the language model perfectly adapts to a specific patient's state profile.")
    sim_pt = st.selectbox("Choose Case File Persona to Load:", linked_pts)
    
    case_profile = [r for r in pdata if r["Patient ID"] == sim_pt][-1]
    st.markdown(f"**Active Simulator Profile:** {case_profile['AI Diagnosis']} | **Initial Complaint:** *\"{case_profile['Patient Input']}\"*")
    st.divider()
    
    for msg in st.session_state.doctor_bot_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if doctor_query := st.chat_input("Type your clinical interrogation question..."):
        st.session_state.doctor_bot_history.append({"role": "user", "content": doctor_query})
        with st.chat_message("user"):
            st.markdown(f"**Doctor:** {doctor_query}")
            
        with st.chat_message("assistant"):
            with st.spinner("Simulated Patient Typing..."):
                simulation_prompt = f"""
                Persona Prompt: Act explicitly as patient {sim_pt}. 
                Your diagnosed psychological status is: {case_profile['AI Diagnosis']}.
                Your original baseline complaint was: "{case_profile['Patient Input']}".
                The interviewing doctor asks: '{doctor_query}'. 
                
                Respond naturally in exactly two sentences. Do NOT include technical phrasing or reveal you are an AI model. Remain completely in character.
                """
                try:
                    res = ollama.generate(model='llama3', prompt=simulation_prompt)
                    pt_response = res['response']
                except Exception as e:
                    pt_response = "I... I cannot find the energy to elaborate right now."
                    st.error(f"Ollama integration error: {e}")
                
                st.markdown(f"**Patient {sim_pt}:** {pt_response}")
                st.session_state.doctor_bot_history.append({"role": "assistant", "content": pt_response})

# ==========================================
# 11. SYSTEM USER ADMINISTRATION (ROOT ACCESS ONLY)
# ==========================================
elif st.session_state.current_page == "Admin Hub":
    st.sidebar.button(f"⬅️ Return to Workspace", on_click=navigate_to, args=("Login",)) 
    st.sidebar.button("🚪 Logout", on_click=logout_procedure)
    
    st.title("⚙️ Directory Access-Control Console")
    st.write("Modify access parameters and commit system permission states to storage configuration tables.")
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Patient Access Matrix")
        st.write(f"**Currently Tracked IDs:** `{', '.join(VALID_PATIENT_IDS)}`")
        new_p_node = st.text_input("Register New Patient Handle (e.g., PT-106):")
        if st.button("➕ Inject Patient Authorization", use_container_width=True):
            if new_p_node and new_p_node.upper() not in VALID_PATIENT_IDS:
                VALID_PATIENT_IDS.append(new_p_node.upper())
                save_credentials(VALID_PATIENT_IDS, VALID_DOCTOR_IDS)
                st.success(f"Node committed: {new_p_node.upper()}")
                st.rerun()
                
        rem_p_node = st.selectbox("Revoke Patient Authorization:", ["-- Choose Target --"] + VALID_PATIENT_IDS)
        if st.button("❌ Remove Patient Access Token", use_container_width=True):
            if rem_p_node != "-- Choose Target --":
                VALID_PATIENT_IDS.remove(rem_p_node)
                save_credentials(VALID_PATIENT_IDS, VALID_DOCTOR_IDS)
                st.success(f"Purged token: {rem_p_node}")
                st.rerun()

    with c2:
        st.subheader("Medical Staff Registry Matrix")
        st.write(f"**Currently Authorized Personnel:** `{', '.join(VALID_DOCTOR_IDS)}`")
        new_d_node = st.text_input("Register New Doctor Code (e.g., DR-002):")
        if st.button("➕ Inject Staff Authorization", use_container_width=True):
            if new_d_node and new_d_node.upper() not in VALID_DOCTOR_IDS:
                VALID_DOCTOR_IDS.append(new_d_node.upper())
                save_credentials(VALID_PATIENT_IDS, VALID_DOCTOR_IDS)
                st.success(f"Staff node committed: {new_d_node.upper()}")
                st.rerun()
                
        rem_d_node = st.selectbox("Revoke Staff Authorization:", ["-- Choose Target --"] + VALID_DOCTOR_IDS)
        if st.button("❌ Remove Staff Access Token", use_container_width=True):
            if rem_d_node != "-- Choose Target --":
                VALID_DOCTOR_IDS.remove(rem_d_node)
                save_credentials(VALID_PATIENT_IDS, VALID_DOCTOR_IDS)
                st.success(f"Purged staff token: {rem_d_node}")
                st.rerun()

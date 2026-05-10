import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import re
import io
from contextlib import redirect_stdout

# ==========================================
# 1. PAGE SETUP & BRANDING
# ==========================================
st.set_page_config(page_title="Anubis AI", page_icon="⚖️", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    div.stButton > button { width: 100%; border-radius: 5px; height: 3em; background-color: #4A4A4A; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚖️ Anubis AI")
st.subheader("Guiding you from Syntax to Strategy")

# ==========================================
# 2. INITIALIZATION (Featherless)
# ==========================================
try:
    client = OpenAI(
        base_url="https://api.featherless.ai/v1",
        api_key=st.secrets["FEATHERLESS_API_KEY"]
    )
except Exception as e:
    st.error("Credential Error: Please check your secrets.toml file.")

MODEL_ID = 'ibm-granite/granite-7b-instruct'

# ==========================================
# 3. CORE LOGIC FUNCTIONS
# ==========================================
def ask_anubis(user_prompt, dynamic_system_prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": dynamic_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=900,
            stop=["[/INST]", "[/SYS]", "Assistant:", "<</SYS>>"] 
        )
        return str(response.choices[0].message.content)
    except Exception as e:
        return f"The Underworld is turbulent: {e}"

# ==========================================
# 4. THE USER INTERFACE & DATA ENGINE
# ==========================================
uploaded_file = st.file_uploader("Drop your data into the Underworld", type=["csv"])

if uploaded_file:
    # 4a. Persistence: Ensure the dataframe is available for the button click
    df_main = pd.read_csv(uploaded_file)
    df_main.to_csv("data.csv", index=False) 
    
    columns_info = ", ".join(df_main.columns.tolist())
    
    with st.expander("👁️ Data Overview"):
        st.write(df_main.head(5))

    user_query = st.text_input("What insight do you seek?", placeholder="e.g., Does garage impact price?")

    if st.button("Query Anubis"):
        # --- 1. THE TRUTH ENGINE ---
        # Include 'bool' to ensure 'garage' is captured in numeric analysis
        numeric_df_raw = df_main.select_dtypes(include=['number', 'bool'])
        
        # FIXED LOGIC: Prioritize 'price' as the target if it exists
        if 'price' in numeric_df_raw.columns:
            target_col = 'price'
        else:
            raw_target = numeric_df_raw.columns[-1]
            target_col = str(raw_target) if not isinstance(raw_target, pd.Index) else str(raw_target[0])
        
        # Clean the data: Drop missing targets and impute others
        numeric_df = numeric_df_raw.dropna(subset=[target_col]).copy()
        for col in numeric_df.columns:
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
        
        full_corr_matrix = numeric_df.corr()

        # Calculate correlations specifically for our target
        corr_matrix = numeric_df.corr()
        correlations = corr_matrix[target_col].drop(labels=[target_col]).to_dict()
        
        # Identify the subject from user query and find the top driver
        matched_col = next((c for c in correlations if c.lower() in user_query.lower()), None)
        top_driver = max(correlations, key=lambda k: abs(correlations[k]))
        
        # Define specific values for the prompt
        subject_val = correlations[matched_col] if matched_col else None
        top_val = correlations[top_driver]

        # --- 2. THE ANALYST PROMPT ---
        DYNAMIC_PROMPT = f"""
        ROLE: Anubis AI (Senior Strategic Analyst).
        
        EVIDENCE:
        - Target Metric: {target_col}
        - Subject Metric: {matched_col if matched_col else 'N/A'} (Correlation: {f"{subject_val:.4f}" if subject_val is not None else "N/A"})
        - Primary Driver: {top_driver} (Correlation: {top_val:.4f})

        MISSION:
        Answer the user's question concisely using ONLY the EVIDENCE above. 
        If the absolute correlation is between -0.1 and 0.1, the answer is "No."
        Explain how the coefficient represents the relationship.
        Direct the user toward the Primary Driver ({top_driver}).

        FORMAT:
        **The Conclusion:** [Explain the coefficient]
        **Strategic Advice:** [Direct focus to {top_driver}]

        STRICT PROTOCOL:
        - DO NOT provide "Confidence Scores", "Summary" sections, or "Meta-commentary".
        - DO NOT repeat your conclusion.
        - STOP writing the moment you finish the Strategic Advice.
        """

        with st.spinner("Anubis is verifying the data truth..."):
            raw_response = ask_anubis(user_query, DYNAMIC_PROMPT)
            clean_answer = str(raw_response).replace("[/INST]", "").replace("[/SYS]", "").strip()
            st.markdown(clean_answer)
            
            with st.expander("👁️ The Universal Evidence Ledger (Full Correlation Matrix)"):
                st.write("This table shows how every variable impacts every other variable. Values closer to 1 or -1 are strong connections.")
                st.dataframe(full_corr_matrix.style.background_gradient(cmap='RdBu', axis=None).format("{:.4f}"))

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("---")
st.caption("Anubis AI | Built for the 2026 Humanitarian Track | Powered by IBM Granite")
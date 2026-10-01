import streamlit as st
import pandas as pd
import json
import re
import base64
from io import BytesIO
import fitz  # PyMuPDF for PDF rendering
from openai import OpenAI

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HK Foundation & Geotechnical AI Analyzer",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ HK Site Investigation AI Analyzer & Foundation Design Assistant")
st.caption("Aligned with HK Code of Practice for Foundations (2017) & GEO Publication No. 1/2006")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_provider = st.selectbox(
        "API Provider",
        ["OpenAI (GPT-4o)", "DeepSeek (DeepSeek-Chat)"],
        index=0
    )
    
    api_key = st.text_input("API Key", type="password")
    
    if api_provider == "DeepSeek (DeepSeek-Chat)":
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-chat"
    else:
        base_url = "https://api.openai.com/v1"
        model_name = "gpt-4o"
        
    st.markdown("---")
    st.subheader("📋 HK COP Foundations (2017) Guidelines")
    st.markdown("""
    * **Category 1(a) Rock**: Grade I/II Rock (qa = 5,000 - 7,500 kPa)
    * **Category 1(b) Rock**: Grade III Rock (qa = 3,000 kPa)
    * **Category 1(c) Rock**: Grade IV / HDG (qa = 750 kPa)
    * **Category 2 Soil**: CDG/CDV (N >= 30, qa = 200 - 300 kPa)
    * **Karst Hazard**: Marble zones require 20 m proof drilling into bedrock.
    """)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_page_numbers(page_str, max_pages):
    """Parses page strings like '1-3, 5, 8' into 0-indexed page list."""
    pages = set()
    parts = page_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                for p in range(int(start), int(end) + 1):
                    if 1 <= p <= max_pages:
                        pages.add(p - 1)
            except ValueError:
                pass
        else:
            try:
                p = int(part)
                if 1 <= p <= max_pages:
                    pages.add(p - 1)
            except ValueError:
                pass
    return sorted(list(pages))

def pdf_pages_to_images(pdf_bytes, selected_pages_0idx, dpi=200):
    """Converts selected PDF pages to base64 encoded PNG images using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    encoded_images = []
    
    for page_num in selected_pages_0idx:
        if page_num < len(doc):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            encoded_images.append((page_num + 1, b64_str))
            
    return encoded_images, len(doc)

def clean_json_response(response_str):
    """Extracts JSON object from LLM response markdown block."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_str)
    if json_match:
        return json_match.group(1).strip()
    return response_str.strip()

SYSTEM_PROMPT = """
You are a Lead Geotechnical & Structural Engineer in Hong Kong specialized in analyzing Site Investigation (SI) borehole logs and evaluating foundation options under the HK Code of Practice for Foundations (2017) and GEO Publication No. 1/2006.

Extract structured data from the provided borehole log image(s) and perform a comprehensive foundation engineering feasibility assessment.

Return EXCLUSIVELY a valid JSON object matching this structure:

{
  "general_info": {
    "borehole_id": "e.g. BH-01",
    "project_name": "String or N/A",
    "ground_level_mPD": "e.g. +6.50 or N/A",
    "groundwater_level_mPD": "e.g. +2.10 or N/A",
    "groundwater_depth_mBGL": "e.g. 4.40 or N/A",
    "termination_depth_mBGL": "e.g. 35.00",
    "date_drilled": "String or N/A"
  },
  "stratigraphy": [
    {
      "depth_from_m": 0.0,
      "depth_to_m": 3.5,
      "geological_description": "FILL: Brown silty fine to coarse sand with gravel",
      "rock_soil_type": "FILL / CDG / HDG / MDG / SDG / Fresh Rock / Marble",
      "decomposition_grade": "Grade VI / V / IV / III / II / I / N/A",
      "spt_n_range": "8 - 15",
      "spt_n_design": 10,
      "presumed_allowable_bearing_pressure_kpa": 100,
      "tcr_percent": "N/A",
      "scr_percent": "N/A",
      "rqd_percent": "N/A"
    }
  ],
  "hk_foundation_design_summary": {
    "top_of_cdg_mPD": "e.g. +3.00 or N/A",
    "top_of_grade_iii_rock_mPD": "e.g. -12.50 or N/A",
    "top_of_grade_ii_i_rock_mPD": "e.g. -18.00 or N/A",
    "category_1a_rock_found": true,
    "category_1b_rock_found": true,
    "max_presumptive_bearing_capacity_kPa": "5000 kPa (Cat 1a)"
  },
  "foundation_feasibility": [
    {
      "foundation_type": "Socketed Steel H-Piles in Rock",
      "feasibility_status": "Highly Recommended / Feasible / Unsuitable",
      "estimated_founding_level_mPD": "-13.5 mPD",
      "estimated_founding_depth_mBGL": "20.0 mBGL",
      "founding_stratum": "Grade III Rock or better (Category 1b)",
      "justification": "Provides high vertical load capacity with minimal settlement. Ideal given moderate rockhead depth."
    },
    {
      "foundation_type": "Large Diameter Bored Piles (LDBP)",
      "feasibility_status": "Highly Recommended / Feasible / Unsuitable",
      "estimated_founding_level_mPD": "-19.0 mPD",
      "estimated_founding_depth_mBGL": "25.5 mBGL",
      "founding_stratum": "Category 1(a) Grade II/I Rock",
      "justification": "Suitable for heavy commercial/residential towers socketed into Category 1(a) bedrock."
    },
    {
      "foundation_type": "Raft / Spread Footings",
      "feasibility_status": "Feasible with Conditions / Unsuitable",
      "estimated_founding_level_mPD": "+2.0 mPD",
      "estimated_founding_depth_mBGL": "4.5 mBGL",
      "founding_stratum": "Dense CDG (SPT N >= 30)",
      "justification": "Feasible for low-rise structures subject to allowable bearing capacity limits (200-300 kPa)."
    },
    {
      "foundation_type": "Driven Steel H-Piles",
      "feasibility_status": "Feasible / Unsuitable",
      "estimated_founding_level_mPD": "-10.0 mPD",
      "estimated_founding_depth_mBGL": "16.5 mBGL",
      "founding_stratum": "Hard CDG / HDG Layer",
      "justification": "May experience refusal on boulders or shallow bedrock. Check vibration restrictions."
    }
  ],
  "geological_hazards_and_anomalies": [
    {
      "hazard_type": "Cavity / Core Loss / Soft Layer / High Water Table / Deep Fill",
      "depth_interval_mBGL": "12.0 - 14.5m",
      "severity": "High / Medium / Low",
      "description": "Drop in RQD to 0% indicating potential fault/fracture zone or karst cavity feature."
    }
  ],
  "recommended_founding_elevations": {
    "shallow_foundation_mPD": "e.g. +3.5 mPD in Dense CDG or N/A",
    "deep_foundation_rockhead_mPD": "e.g. -12.5 mPD (Cat 1b) / -18.0 mPD (Cat 1a)",
    "pre_drilling_recommendation": "100% pre-drilling required for socketed H-piles / 20m bedrock proof drilling if marble encountered."
  }
}

Guidelines for Assessment:
1. Bearing Pressures follow HK COP Foundations (2017) Table 5.1:
   - Soil/Fill (N < 10): 0 kPa (unsuitable).
   - CDG (10 <= N < 30): 100 kPa.
   - Dense CDG (N >= 30): 200 - 300 kPa.
   - Grade IV / HDG (N >= 100): 750 kPa (Cat 1c).
   - Grade III Rock: 3,000 kPa (Cat 1b).
   - Grade II/I Rock: 5,000 - 7,500 kPa (Cat 1a).
2. Scan for anomalies: sudden loss of flush water, core recovery < 10%, cavities, marble, soft clay / marine deposit layers subject to dragdown (negative skin friction).
3. Recommend feasible foundation types tailored to the specific ground conditions detected.
"""

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.markdown("### 📄 Upload Borehole Logs / SI Reports")

col_file, col_pages = st.columns([2, 1])

with col_file:
    uploaded_files = st.file_uploader(
        "Upload PDF report or image files (PNG/JPG)", 
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

with col_pages:
    page_range_input = st.text_input(
        "Drillhole Log Pages to Scan",
        value="1-3",
        help="Specify PDF page numbers where drillhole logs are located (e.g. '1-3, 5, 8')."
    )

if st.button("🚀 Analyze SI Report & Feasibility", type="primary"):
    if not api_key:
        st.warning("Please enter your API Key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.warning("Please upload at least one borehole log file.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)
    msg_content = [{"type": "text", "text": "Analyze this drillhole log for HK foundation design. Evaluate foundation feasibility, detect geological hazards/cavities, and recommend founding levels."}]

    with st.spinner("Processing PDF pages and evaluating foundation parameters..."):
        for file in uploaded_files:
            file_bytes = file.read()
            file_ext = file.name.split('.')[-1].lower()

            if file_ext == 'pdf':
                doc_temp = fitz.open(stream=file_bytes, filetype="pdf")
                total_pages = len(doc_temp)
                selected_pages = parse_page_numbers(page_range_input, total_pages)

                if not selected_pages:
                    st.error(f"No valid pages specified for '{file.name}'. Total pages: {total_pages}")
                    st.stop()

                st.info(f"Scanning PDF '{file.name}' — Pages: {[p+1 for p in selected_pages]} of {total_pages}")
                encoded_imgs, _ = pdf_pages_to_images(file_bytes, selected_pages, dpi=200)

                for page_num, b64_img in encoded_imgs:
                    msg_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_img}",
                            "detail": "high"
                        }
                    })

            elif file_ext in ['png', 'jpg', 'jpeg', 'webp']:
                encoded_img = base64.b64encode(file_bytes).decode('utf-8')
                msg_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{file_ext};base64,{encoded_img}",
                        "detail": "high"
                    }
                })

        # Call API
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg_content}
                ]
            )

            raw_result = response.choices[0].message.content
            clean_json = clean_json_response(raw_result)
            parsed_data = json.loads(clean_json)

            st.success("Analysis & Feasibility Assessment Completed Successfully!")

            # -----------------------------------------------------------------
            # Extract Data Structures
            # -----------------------------------------------------------------
            gen_info = parsed_data.get("general_info", {})
            hk_summary = parsed_data.get("hk_foundation_design_summary", {})
            strat_list = parsed_data.get("stratigraphy", [])
            feasibility_list = parsed_data.get("foundation_feasibility", [])
            hazards_list = parsed_data.get("geological_hazards_and_anomalies", [])
            founding_elev = parsed_data.get("recommended_founding_elevations", {})

            # Summary Metrics Bar
            st.markdown("### 📊 Key Geological Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Borehole ID", gen_info.get("borehole_id", "N/A"))
            m2.metric("Ground Level", gen_info.get("ground_level_mPD", "N/A"))
            m3.metric("Water Level (mPD)", gen_info.get("groundwater_level_mPD", "N/A"))
            m4.metric("Water Depth (mBGL)", gen_info.get("groundwater_depth_mBGL", "N/A"))
            m5.metric("Rockhead (Cat 1b)", hk_summary.get("top_of_grade_iii_rock_mPD", "N/A"))

            st.markdown("---")

            # Workspace Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "💡 Foundation Feasibility & Founding Levels", 
                "⚠️ Geological Hazards & Weak Zones",
                "📋 Stratigraphy & SPT Table", 
                "🏛️ HK COP Foundations (2017) Criteria", 
                "🔍 Raw JSON Output"
            ])

            # -----------------------------------------------------------------
            # TAB 1: Foundation Feasibility
            # -----------------------------------------------------------------
            with tab1:
                st.subheader("💡 Foundation Option Analysis & Recommended Levels")
                
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.info(f"**Recommended Shallow Founding Elevation:**\n\n`{founding_elev.get('shallow_foundation_mPD', 'N/A')}`")
                with col_e2:
                    st.success(f"**Recommended Deep Founding Elevation (Rockhead):**\n\n`{founding_elev.get('deep_foundation_rockhead_mPD', 'N/A')}`")

                st.markdown("#### Foundation Feasibility Comparison")
                if feasibility_list:
                    df_feas = pd.DataFrame(feasibility_list)
                    st.dataframe(
                        df_feas,
                        use_container_width=True,
                        column_config={
                            "foundation_type": "Foundation Type",
                            "feasibility_status": "Status",
                            "estimated_founding_level_mPD": "Founding Level (mPD)",
                            "estimated_founding_depth_mBGL": "Depth (mBGL)",
                            "founding_stratum": "Target Stratum",
                            "justification": "Engineering Justification"
                        }
                    )
                
                st.markdown("#### 📌 Key Construction & Drilling Recommendations")
                st.write(founding_elev.get("pre_drilling_recommendation", "Standard pre-drilling procedures apply."))

            # -----------------------------------------------------------------
            # TAB 2: Geological Hazards & Anomalies
            # -----------------------------------------------------------------
            with tab2:
                st.subheader("⚠️️ Detected Hazards, Cavities & Weak Zones")
                if hazards_list:
                    for haz in hazards_list:
                        sev = str(haz.get("severity", "")).lower()
                        box_title = f"**Hazard:** {haz.get('hazard_type', 'Geological Feature')} | **Depth:** `{haz.get('depth_interval_mBGL', 'N/A')}` | **Severity:** {haz.get('severity', 'Medium')}"
                        
                        if "high" in sev:
                            st.error(f"🚨 {box_title}\n\n{haz.get('description', '')}")
                        elif "medium" in sev:
                            st.warning(f"⚠️ {box_title}\n\n{haz.get('description', '')}")
                        else:
                            st.info(f"ℹ️ {box_title}\n\n{haz.get('description', '')}")
                else:
                    st.success("✅ No severe cavities, fault zones, or critical geological anomalies detected in this borehole log.")

            # -----------------------------------------------------------------
            # TAB 3: Stratigraphy & SPT Table
            # -----------------------------------------------------------------
            with tab3:
                st.subheader("Borehole Stratigraphy & In-Situ Test Results")
                if strat_list:
                    df_strat = pd.DataFrame(strat_list)
                    col_order = [
                        "depth_from_m", "depth_to_m", "geological_description", 
                        "rock_soil_type", "decomposition_grade", "spt_n_range", 
                        "spt_n_design", "presumed_allowable_bearing_pressure_kpa",
                        "tcr_percent", "scr_percent", "rqd_percent"
                    ]
                    available_cols = [c for c in col_order if c in df_strat.columns]
                    df_strat = df_strat[available_cols]

                    st.dataframe(
                        df_strat, 
                        use_container_width=True,
                        column_config={
                            "depth_from_m": st.column_config.NumberColumn("From (m)", format="%.2f"),
                            "depth_to_m": st.column_config.NumberColumn("To (m)", format="%.2f"),
                            "geological_description": "Geological Description",
                            "rock_soil_type": "Soil/Rock Type",
                            "decomposition_grade": "Grade",
                            "spt_n_range": "SPT N Range",
                            "spt_n_design": st.column_config.NumberColumn("SPT N"),
                            "presumed_allowable_bearing_pressure_kpa": st.column_config.NumberColumn("Bearing Cap. (kPa)"),
                            "tcr_percent": "TCR (%)",
                            "scr_percent": "SCR (%)",
                            "rqd_percent": "RQD (%)"
                        }
                    )

                    csv_data = df_strat.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Take-off to CSV",
                        data=csv_data,
                        file_name=f"{gen_info.get('borehole_id', 'borehole')}_takeoff.csv",
                        mime="text/csv"
                    )

            # -----------------------------------------------------------------
            # TAB 4: HK COP Foundations Criteria
            # -----------------------------------------------------------------
            with tab4:
                st.subheader("Hong Kong COP Foundations (2017) Rockhead Assessment")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Strata Elevation Horizons")
                    st.write(f"**Top of CDG Layer:** `{hk_summary.get('top_of_cdg_mPD', 'N/A')}`")
                    st.write(f"**Top of Grade III Rock (Cat 1b):** `{hk_summary.get('top_of_grade_iii_rock_mPD', 'N/A')}`")
                    st.write(f"**Top of Grade II/I Bedrock (Cat 1a):** `{hk_summary.get('top_of_grade_ii_i_rock_mPD', 'N/A')}`")

                with c2:
                    st.markdown("#### Rock Classification Summary")
                    st.write(f"**Category 1(a) Rock Available:** {'✅ Yes' if hk_summary.get('category_1a_rock_found') else '❌ No'}")
                    st.write(f"**Category 1(b) Rock Available:** {'✅ Yes' if hk_summary.get('category_1b_rock_found') else '❌ No'}")
                    st.write(f"**Max Presumptive Capacity:** `{hk_summary.get('max_presumptive_bearing_capacity_kPa', 'N/A')}`")

            # -----------------------------------------------------------------
            # TAB 5: Raw JSON
            # -----------------------------------------------------------------
            with tab5:
                st.json(parsed_data)

        except Exception as e:
            st.error(f"Error executing API request: {e}")
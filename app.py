from pathlib import Path
import tempfile
import json
import io

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from src.predictor import AnomalyPredictor


# ==========================================
# 0. HTML HELPER
# ==========================================

def render_html(html: str):
    """
    Render raw HTML safely through st.markdown.

    Markdown treats any line indented by 4+ spaces that follows a blank
    line as a code block, which is why tags like <br> were shown as text.
    Stripping indentation and blank lines from every line avoids this.
    """
    cleaned = "\n".join(
        line.strip() for line in html.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="InspectAI | Industrial Quality Control",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

render_html("""
<style>

    /* Main application background */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.9);
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Main heading */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Dashboard cards */
    .info-card {
        background: linear-gradient(135deg, #172337, #111c2e);
        border: 1px solid #293b53;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .info-card-title {
        color: #38bdf8;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .info-card-text {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0 0 10px 0;
    }

    .result-card {
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .pass-card {
        background: linear-gradient(120deg, #063b32, #0d2c2b);
        border: 1px solid #10b981;
    }

    .fail-card {
        background: linear-gradient(120deg, #491d27, #301923);
        border: 1px solid #ef4444;
    }

    .result-label {
        font-size: 0.85rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-top: 5px;
    }

    .result-description {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 5px;
    }

    /* Small labels */
    .muted-text {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        border: 1px solid #60a5fa;
        color: white;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #111c2e;
        border: 1px solid #293b53;
        padding: 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Divider */
    hr {
        border-color: #293b53;
    }

    /* Footer */
    .footer {
        color: #64748b;
        text-align: center;
        font-size: 0.8rem;
        padding-top: 30px;
    }

    .footer-line {
        margin-bottom: 14px;
    }

</style>
""")


# ==========================================
# 3. LOAD PATCHCORE MODEL
# ==========================================

@st.cache_resource
def load_predictor():

    project_root = Path(__file__).resolve().parent

    return AnomalyPredictor(project_root)


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🔍 InspectAI")

    st.caption("INDUSTRIAL QUALITY CONTROL")

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
**Architecture:** PatchCore

**Dataset:** MVTec AD

**Category:** Bottle

**Framework:** PyTorch + Anomalib
""")

    st.divider()

    st.markdown("### Inspection Process")

    st.markdown("""
1. Upload a bottle image.
2. Run anomaly detection.
3. Review the anomaly score.
4. Inspect the defect heatmap.
5. Download the inspection report.
""")

    st.divider()

    st.caption("Developed as a Computer Vision portfolio project.")


# ==========================================
# 5. MAIN HEADER
# ==========================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:

    render_html("""
    <div class="main-title">Industrial Anomaly Detection</div>
    <div class="subtitle">
        AI-powered visual inspection for automated product quality control.
    </div>
    """)

with header_col2:

    render_html("""
    <div style="text-align:right; padding-top:12px;">
        <span style="background:#064e3b; color:#6ee7b7;
        padding:9px 14px; border-radius:20px; font-size:13px;">
            ● PATCHCORE
        </span>
    </div>
    """)


# ==========================================
# 6. INITIALIZE MODEL
# ==========================================

try:

    predictor = load_predictor()

    st.success("PatchCore model is ready for inspection.")

except Exception as error:

    st.error(f"Model initialization failed: {error}")
    st.stop()


# ==========================================
# 7. IMAGE UPLOAD SECTION
# ==========================================

render_html("""
<div class="section-heading">Image Inspection</div>
""")

upload_col, instruction_col = st.columns([1.5, 1])

with upload_col:

    uploaded_file = st.file_uploader(
        "Upload a bottle image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG and JPEG"
    )

with instruction_col:

    render_html("""
    <div class="info-card">
        <div class="info-card-title">INSPECTION GUIDE</div>
        <p class="info-card-text">
            Upload a clear image of a bottle.
        </p>
        <p class="info-card-text">
            The PatchCore model will analyze the image
            and identify potentially defective regions.
        </p>
    </div>
    """)


# ==========================================
# 8. RUN PREDICTION
# ==========================================

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    render_html("""
    <div class="section-heading">Input Preview</div>
    """)

    preview_col, action_col = st.columns([1, 2])

    with preview_col:

        st.image(
            original_image,
            caption=uploaded_file.name,
            width="stretch"
        )

    with action_col:

        render_html("""
        <div class="info-card">
            <div class="info-card-title">READY FOR ANALYSIS</div>
            <div class="info-card-text">
                Your image has been uploaded successfully.
                Start the inspection to generate the prediction
                and defect localization results.
            </div>
        </div>
        """)

        analyze_button = st.button(
            "▶  Analyze Image",
            type="primary"
        )

    if analyze_button:

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            with st.spinner(
                "PatchCore is analyzing the uploaded image..."
            ):

                result = predictor.predict(temp_path)

            # Extract results
            anomaly_score = result["anomaly_score"]
            threshold = result["threshold"]
            predicted_class = result["predicted_class"]

            is_defective = predicted_class == "Defective"

            # ==================================
            # 9. INSPECTION RESULT
            # ==================================

            st.divider()

            render_html("""
            <div class="section-heading">Inspection Results</div>
            """)

            if is_defective:

                render_html("""
                <div class="result-card fail-card">
                    <div class="result-label">INSPECTION STATUS</div>
                    <div class="result-value">⚠ FAIL — DEFECT DETECTED</div>
                    <div class="result-description">
                        The model identified anomalous visual patterns
                        in the uploaded product.
                    </div>
                </div>
                """)

            else:

                render_html("""
                <div class="result-card pass-card">
                    <div class="result-label">INSPECTION STATUS</div>
                    <div class="result-value">✓ PASS — NORMAL PRODUCT</div>
                    <div class="result-description">
                        The image-level anomaly score is below
                        the configured classification threshold.
                    </div>
                </div>
                """)

            # ==================================
            # 10. METRICS
            # ==================================

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Anomaly Score",
                    f"{anomaly_score:.4f}"
                )

            with metric2:

                st.metric(
                    "Classification Threshold",
                    f"{threshold:.4f}"
                )

            with metric3:

                st.metric(
                    "Prediction",
                    predicted_class
                )

            st.markdown("**Anomaly score scale**")

            st.progress(
                max(0.0, min(float(anomaly_score), 1.0))
            )

            st.caption(
                "This is a normalized anomaly score, not a probability "
                "or model confidence percentage."
            )

            # ==================================
            # 11. LOCALIZATION
            # ==================================

            st.divider()

            render_html("""
            <div class="section-heading">Defect Localization</div>
            """)

            anomaly_map = result["anomaly_map"]

            defect_mask = result["defect_mask"]

            mask_image = Image.fromarray(
                (defect_mask > 0).astype(np.uint8) * 255
            )

            # Create heatmap image
            heatmap_buffer = io.BytesIO()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.imshow(anomaly_map, cmap="jet")
            ax.axis("off")

            plt.tight_layout(pad=0)

            fig.savefig(
                heatmap_buffer,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=160
            )

            plt.close(fig)

            heatmap_buffer.seek(0)

            # Three visualization columns
            image_col, heatmap_col, mask_col = st.columns(3)

            with image_col:

                st.markdown("**Original Image**")

                st.image(
                    original_image,
                    width="stretch"
                )

            with heatmap_col:

                st.markdown("**Anomaly Heatmap**")

                st.image(
                    heatmap_buffer,
                    width="stretch"
                )

            with mask_col:

                st.markdown("**Predicted Defect Mask**")

                st.image(
                    mask_image,
                    width="stretch"
                )

            # ==================================
            # 12. DOWNLOAD RESULTS
            # ==================================

            st.divider()

            render_html("""
            <div class="section-heading">Export Inspection Results</div>
            """)

            report = {
                "image": uploaded_file.name,
                "model": "PatchCore",
                "dataset": "MVTec AD",
                "category": "bottle",
                "anomaly_score": float(anomaly_score),
                "classification_threshold": float(threshold),
                "predicted_class": predicted_class,
                "inspection_status": "FAIL" if is_defective else "PASS"
            }

            report_json = json.dumps(
                report,
                indent=4
            )

            download_col1, download_col2 = st.columns(2)

            with download_col1:

                st.download_button(
                    "⬇ Download Inspection Report",
                    data=report_json,
                    file_name="inspection_report.json",
                    mime="application/json",
                    use_container_width=True
                )

            with download_col2:

                mask_buffer = io.BytesIO()
                mask_image.save(mask_buffer, format="PNG")

                st.download_button(
                    "⬇ Download Defect Mask",
                    data=mask_buffer.getvalue(),
                    file_name="predicted_defect_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as error:

            st.error(f"Prediction failed: {error}")

        finally:

            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

else:

    st.info(
        "Upload a bottle image above to start the inspection."
    )


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

render_html("""
<div class="footer">
    <div class="footer-line">INSPECTAI · INDUSTRIAL ANOMALY DETECTION</div>
    <div>Powered by PatchCore · PyTorch · Anomalib · Streamlit</div>
</div>
""")
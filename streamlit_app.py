"""Biometric Authentication Demo - Streamlit app.

One app, five pages (sidebar navigation):
  Home              - real face verification (register + verify with cosine sim)
  Feature Extraction     - Student 1
  Template Protection    - Student 2
  Deep Models            - Student 3
  Attacks & Liveness     - Student 4

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Biometric Auth Demo", layout="wide",
                   page_icon="🔐")

# ---------------------------------------------------------------------------
# Premium Custom Styling & Effects
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* Root Variables */
        :root {
            --bg-dark: #05070D;
            --card-bg: #0B0F19;
            --purple-start: #6C63FF;
            --purple-end: #8E44FF;
            --green-accent: #00C853;
            --text-light: #E2E8F0;
            --text-muted: #94A3B8;
        }

        /* Apply Outfit Font globally */
        html, body, [class*="css"], .stMarkdown, p, span, label, input, button {
            font-family: 'Outfit', sans-serif !important;
        }

        /* App Background */
        .stApp {
            background-color: var(--bg-dark) !important;
            color: var(--text-light) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #080B12 !important;
            border-right: 1px solid rgba(108, 99, 255, 0.15) !important;
        }
        
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-light) !important;
        }

        /* Custom Card/Columns Styling */
        div[data-testid="column"] {
            background: var(--card-bg) !important;
            border: 1px solid rgba(108, 99, 255, 0.15) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            position: relative !important;
            overflow: hidden !important;
        }

        /* Lift cards 6px, add glowing shadow and dynamic border color on hover */
        div[data-testid="column"]:hover {
            transform: translateY(-6px) !important;
            border-color: var(--purple-start) !important;
            box-shadow: 0 16px 40px rgba(108, 99, 255, 0.25), 0 0 20px rgba(108, 99, 255, 0.1) !important;
        }

        /* Accent green custom styles */
        .glow-green {
            box-shadow: 0 0 15px rgba(0, 200, 83, 0.3) !important;
            border-color: var(--green-accent) !important;
        }

        /* Premium Buttons with Gradient & Glow */
        div.stButton > button {
            background: linear-gradient(135deg, var(--purple-start) 0%, var(--purple-end) 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            width: auto !important;
        }

        div.stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(142, 68, 255, 0.5), 0 0 15px rgba(108, 99, 255, 0.3) !important;
            background: linear-gradient(135deg, var(--purple-end) 0%, var(--purple-start) 100%) !important;
            color: #FFFFFF !important;
        }

        div.stButton > button:active {
            transform: translateY(1px) !important;
        }

        /* File Uploader Custom Styling */
        div[data-testid="stFileUploader"] {
            background: #080B12 !important;
            border: 2px dashed rgba(108, 99, 255, 0.25) !important;
            border-radius: 14px !important;
            padding: 20px !important;
            transition: all 0.3s ease !important;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: var(--purple-start) !important;
            background: #0c0f1a !important;
            box-shadow: 0 0 15px rgba(108, 99, 255, 0.15) !important;
        }

        /* Metrics Styling - Glow Green & Gradients */
        div[data-testid="stMetricValue"] {
            font-weight: 800 !important;
            background: linear-gradient(135deg, var(--green-accent) 0%, var(--purple-start) 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 2.2rem !important;
        }

        /* Tabs Selection Styling */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-muted) !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            padding: 10px 20px !important;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--text-light) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF !important;
            border-bottom: 3px solid var(--purple-end) !important;
            background: linear-gradient(0deg, rgba(142, 68, 255, 0.08) 0%, rgba(142, 68, 255, 0) 100%) !important;
        }

        /* Inputs & Dropdowns */
        div[data-baseweb="select"], div[data-baseweb="input"], input {
            background-color: #080B12 !important;
            border: 1px solid rgba(108, 99, 255, 0.2) !important;
            border-radius: 10px !important;
            color: var(--text-light) !important;
        }

        /* Success & Info boxes styled with matching gradients */
        div[class*="stAlert"] {
            background-color: #091311 !important;
            border: 1px solid rgba(0, 200, 83, 0.2) !important;
            border-left: 5px solid var(--green-accent) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0, 200, 83, 0.05) !important;
        }

        div[class*="stAlert"]:has([class*="stIconError"]) {
            background-color: #190B0B !important;
            border: 1px solid rgba(255, 51, 51, 0.2) !important;
            border-left: 5px solid #FF3333 !important;
            box-shadow: 0 4px 15px rgba(255, 51, 51, 0.05) !important;
        }
        
        /* Custom card class for manual wrap */
        .premium-info-card {
            background: linear-gradient(135deg, #0B0F19 0%, #111625 100%) !important;
            border: 1px solid rgba(108, 99, 255, 0.15) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin: 16px 0 !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        .premium-info-card:hover {
            transform: translateY(-6px) !important;
            border-color: var(--purple-start) !important;
            box-shadow: 0 16px 40px rgba(108, 99, 255, 0.2), 0 0 20px rgba(108, 99, 255, 0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

PAGES = [
    "Home - Live Verification",
    "Feature Extraction (S1)",
    "Template Protection (S2)",
    "Deep Models (S3)",
    "Attacks & Liveness (S4)",
]


def sidebar():
    st.sidebar.title("Biometric Authentication")
    st.sidebar.caption("A working demo - not a mockup")
    page = st.sidebar.radio("Module", PAGES)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Students**\n"
        "1. Feature Extraction\n"
        "2. Template Protection\n"
        "3. Deep Models\n"
        "4. Attacks & Liveness")
    return page


# ---------------------------------------------------------------------------
# Home - real face verification (the bug, fixed)
# ---------------------------------------------------------------------------

def page_home():
    st.title("Home - Real Face Verification")
    st.markdown(
        "This page actually verifies a face against a stored template using "
        "FaceNet embeddings and cosine similarity. **This is the part the old "
        "app faked** - here the score is real.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Enrol")
        enroll_img = st.camera_input("Capture enrolment face", key="enroll")
        if st.button("Store my face template") and enroll_img is not None:
            img = Image.open(enroll_img)
            try:
                from utils.face import get_face_embedding
                emb, t = get_face_embedding(img)
                st.session_state["enrolled_embedding"] = emb
                st.session_state["enrolled_image"] = img
                st.success(f"Template stored - 512-D embedding, {t:.2f}s")
            except Exception as e:
                st.error(f"Could not extract embedding: {e}")
        if "enrolled_embedding" in st.session_state:
            st.caption("Status: template enrolled")

    with col2:
        st.subheader("2. Verify")
        verify_img = st.camera_input("Capture verification face", key="verify")
        if st.button("Verify identity") and verify_img is not None:
            if "enrolled_embedding" not in st.session_state:
                st.warning("Enrol a face first.")
            else:
                img = Image.open(verify_img)
                try:
                    from utils.face import get_face_embedding, cosine_similarity
                    q_emb, t = get_face_embedding(img)
                    score = cosine_similarity(q_emb, st.session_state["enrolled_embedding"])
                    matched = score >= 0.6
                    st.metric("Cosine similarity", f"{score:.4f}")
                    if matched:
                        st.success(f"MATCH - identity verified (score {score:.4f} >= 0.6)")
                    else:
                        st.error(f"NO MATCH - different face (score {score:.4f} < 0.6)")
                    st.caption(f"Inference time: {t:.2f}s. Threshold: 0.6 (tune on a dataset for real use.)")
                except Exception as e:
                    st.error(f"Could not verify: {e}")

    st.markdown(
        """
        <div class="premium-info-card">
            <h3>Why this matters</h3>
            <p>In the original project, the 'Face Authentication' page waited 2 seconds 
            and showed a hardcoded <code>98.7%</code> score regardless of the image. The <code>/login</code> 
            endpoint only checked username+password and never actually compared faces.</p>
            <p>Here, the cosine similarity is computed from a real FaceNet embedding 
            every time - upload a different person's face and the score drops accordingly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Student 1 - Feature Extraction
# ---------------------------------------------------------------------------

def page_feature_extraction():
    st.title("Student 1 - Feature Extraction Techniques")
    st.markdown("Handcrafted vs deep features on fingerprint and face.")

    tab_fp, tab_face, tab_table = st.tabs(["Fingerprint", "Face (deep)", "Comparison"])

    with tab_fp:
        st.subheader("Fingerprint feature extraction")
        fp_file = st.file_uploader("Upload a fingerprint image",
                                   type=["png", "jpg", "jpeg"], key="fp")
        use_sample = st.checkbox("Use a sample fingerprint if you have none",
                                 value=True)
        img = None
        if fp_file is not None:
            img = Image.open(fp_file).convert("L")
        elif use_sample:
            from pathlib import Path
            samples = list(Path("data/fingerprints").glob("*"))
            if samples:
                img = Image.open(samples[0]).convert("L")
                st.caption(f"Using sample: {samples[0].name}")
            else:
                st.info("No sample fingerprints in data/fingerprints/. Upload one.")
        if img is not None:
            import cv2
            from modules.feature_extraction import (
                gabor_enhance, extract_minutiae, draw_minutiae,
                lbp_histogram, sift_keypoints)
            from utils.viz import image_grid
            gray = np.array(img)
            st.image(gray, caption="Original", use_column_width=True, clamp=True)
            with st.spinner("Extracting features..."):
                enhanced, responses = gabor_enhance(gray)
                minutiae, skel, bin_img = extract_minutiae(gray)
                hist, lbp_img = lbp_histogram(gray)
                kp, sift_img = sift_keypoints(gray)
            c1, c2 = st.columns(2)
            with c1:
                st.image(enhanced, caption="Gabor enhanced", clamp=True)
                st.image(draw_minutiae(gray, minutiae),
                         caption=f"Minutiae ({len(minutiae)}: "
                                 f"{sum(1 for m in minutiae if m['type']=='ending')} endings, "
                                 f"{sum(1 for m in minutiae if m['type']=='bifurcation')} bifurcations)",
                         clamp=True)
            with c2:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(range(len(hist)), hist, color="#4C3FE4")
                ax.set_title("LBP histogram"); ax.set_xlabel("bin")
                fig.tight_layout()
                st.pyplot(fig)
                st.image(sift_img, caption=f"SIFT keypoints ({len(kp)})", clamp=True)

    with tab_face:
        st.subheader("Face deep embedding (FaceNet)")
        face_file = st.file_uploader("Upload a face image",
                                     type=["png", "jpg", "jpeg"], key="fe_face")
        if face_file is not None:
            from modules.feature_extraction import face_deep_embedding
            from utils.viz import embedding_bar_chart, embedding_heatmap
            img = Image.open(face_file)
            st.image(img, caption="Input", width=200)
            try:
                with st.spinner("Extracting FaceNet embedding..."):
                    emb, t = face_deep_embedding(img)
                st.success(f"512-D embedding extracted in {t:.2f}s")
                st.pyplot(embedding_bar_chart(emb, "FaceNet embedding"))
                st.pyplot(embedding_heatmap(emb))
            except Exception as e:
                st.error(f"Could not extract face embedding: {e}")

    with tab_table:
        from modules.feature_extraction import feature_comparison_table
        st.markdown("### Handcrafted vs deep - methods demonstrated here")
        st.dataframe(feature_comparison_table(), use_container_width=True)
        st.markdown(
            """
            <div class="premium-info-card">
                <h4>📚 Survey Task (Student 1)</h4>
                <p>Expand this into a 15+ paper table with columns: <strong>dataset, method, accuracy, advantages, limitations</strong>.</p>
                <p><em>Seed references:</em> minutiae (NIST NBIS), Gabor (Hong et al. 1998), LBP (Ahonen et al. 2006), SIFT (Lowe 2004), FaceNet (Schroff 2015), DeepPrint (Engelsma 2021), FingerNet (Tang 2017), pyfing/SNFEN (Cappelli 2025).</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Student 2 - Template Protection
# ---------------------------------------------------------------------------

def page_template_protection():
    st.title("Student 2 - Biometric Template Protection")
    st.markdown(
        "Cancel your face template: apply keyed transforms so the stored "
        "template is non-invertible, revocable, and unlinkable. "
        "Schemes benchmarked in Otroshi et al. (2025, Springer).")

    face_file = st.file_uploader("Upload a face image to derive the embedding",
                                 type=["png", "jpg", "jpeg"], key="tp_face")
    if face_file is None:
        st.info("Upload a face image to begin.")
        return
    img = Image.open(face_file)
    st.image(img, caption="Input face", width=180)

    from utils.face import get_face_embedding
    try:
        with st.spinner("Extracting FaceNet embedding..."):
            emb, _ = get_face_embedding(img)
    except Exception as e:
        st.error(f"Face embedding failed: {e}")
        return

    seed = st.number_input("Protection key (seed)", value=42, step=1)
    scheme = st.selectbox(
        "Scheme",
        ["BioHashing", "IoM-URP", "Bloom filter", "Fuzzy commitment",
         "Chaotic-map scrambling"])

    from modules import template_protection as tp

    if st.button("Apply protection"):
        s = int(seed)
        if scheme == "BioHashing":
            tmpl = tp.biohash(emb, s)
            st.write(f"Protected template (256-bit): `{tmpl[:32].tolist()}...`")
        elif scheme == "IoM-URP":
            tmpl = tp.iom_urp(emb, s)
            st.write(f"Protected template (indices): `{tmpl.tolist()}`")
        elif scheme == "Bloom filter":
            tmpl = tp.bloom_template(emb, s)
            st.write(f"Protected template (bloom codes, {len(tmpl)} bits): "
                     f"`{tmpl[:32].tolist()}...`")
        elif scheme == "Fuzzy commitment":
            secret = b"BIO-SECRET-2026"
            fc = tp.fuzzy_commit(emb, secret)
            st.write(f"Bound secret: `{secret.decode()}`")
            st.write(f"Helper data (XOR bits, {len(fc['helper'])}): "
                     f"`{fc['helper'][:32].tolist()}...`")
            # demo decode with same embedding -> should recover
            ok, dec = tp.fuzzy_commit_decode(emb, fc)
            st.write(f"Self-decode (same face): recovered=`{dec}` ok={ok}")
        elif scheme == "Chaotic-map scrambling":
            tmpl = tp.chaotic_scramble(emb, s)
            import matplotlib.pyplot as plt
            from utils.viz import embedding_heatmap
            st.pyplot(embedding_heatmap(emb, "Original embedding"))
            st.pyplot(embedding_heatmap(tmpl, "Scrambled embedding"))

    st.markdown("---")
    st.subheader("Revocability demo (same face, different keys)")
    if st.button("Run revocability demo"):
        df = tp.revocability_demo(emb, scheme="biohash", n_keys=5)
        st.dataframe(df, use_container_width=True)
        st.markdown(
            f"Mean inter-key Hamming distance: **{df['hamming_distance'].mean():.3f}** "
            "(~0.5 means different keys -> uncorrelated templates = revocable & unlinkable)")

    st.markdown("---")
    st.subheader("Comparison of schemes")
    st.dataframe(tp.protection_comparison_table(), use_container_width=True)
    st.markdown(
        """
        <div class="premium-info-card">
            <h4>📚 Survey Task (Student 2)</h4>
            <p>Expand to 15-20 papers. <em>Seed refs:</em> BioHashing (Teoh 2004), IoM (Jin 2018), Bloom (Rathgeb 2013), Fuzzy commitment (Juels 2002), Fuzzy vault (Juels 2002), Homomorphic encryption for biometrics, Otroshi et al. 2025 (Springer benchmark), BioDeepHash (IEEE TDSC 2026), ISO/IEC 24745.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Student 3 - Deep Models
# ---------------------------------------------------------------------------

def page_deep_models():
    st.title("Student 3 - Deep Learning Models Comparison")
    st.markdown(
        "Live measurement of parameters, FLOPs, inference time and cosine "
        "similarity for three runnable face-embedding backbones.")

    c1, c2 = st.columns(2)
    with c1:
        fa = st.file_uploader("Face A (same person preferred)",
                              type=["png", "jpg", "jpeg"], key="dm_a")
    with c2:
        fb = st.file_uploader("Face B (same person as A, or impostor)",
                              type=["png", "jpg", "jpeg"], key="dm_b")
    if fa is None or fb is None:
        st.info("Upload two face images to compare the models.")
        return
    img_a, img_b = Image.open(fa), Image.open(fb)
    c1.image(img_a, caption="Face A", width=160)
    c2.image(img_b, caption="Face B", width=160)

    from modules.deep_models import run_comparison, landscape_table
    if st.button("Run live comparison"):
        with st.spinner("Running FaceNet, MobileNetV2, ResNet18... (first run downloads weights)"):
            res = run_comparison(img_a, img_b)
        df = pd.DataFrame(res["results"])
        st.dataframe(df, use_container_width=True)
        import matplotlib.pyplot as plt
        ok = df[df["cosine_similarity"].notna()] if "cosine_similarity" in df else df
        if not ok.empty and "params" in ok:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            axes[0].bar(ok["model"], ok["params"] / 1e6); axes[0].set_title("Params (M)"); axes[0].tick_params(axis='x', rotation=20)
            axes[1].bar(ok["model"], ok["flops"] / 1e9); axes[1].set_title("FLOPs (G)"); axes[1].tick_params(axis='x', rotation=20)
            axes[2].bar(ok["model"], ok["cosine_similarity"]); axes[2].set_title("Cosine sim A vs B"); axes[2].tick_params(axis='x', rotation=20)
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("2026 landscape (cited - for the survey table)")
    st.dataframe(landscape_table(), use_container_width=True)
    st.markdown(
        """
        <div class="premium-info-card">
            <h4>📚 Survey Task (Student 3)</h4>
            <p>Expand to 15+ papers with FLOPs, params, accuracy, inference time. <em>Seed refs:</em> FaceNet (Schroff 2015), ArcFace (Deng 2019), AdaFace (Kim 2022), MagFace (Meng 2021), EdgeFace (Boutros 2023), MobileNet (Howard 2017), ViT (Dosovitskiy 2021), NPTFace (CVPR 2026), FunFace (2026).</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Student 4 - Attacks & Liveness
# ---------------------------------------------------------------------------

def page_attacks_liveness():
    st.title("Student 4 - Attacks & Liveness Detection")
    st.markdown("Compare a real lightweight deep liveness model (MiniFASNet) "
                "against an easily-fooled heuristic baseline.")

    tab_live, tab_attack, tab_table = st.tabs(["Liveness test", "Attack taxonomy", "Comparison"])

    with tab_live:
        face_file = st.file_uploader("Upload a face image (real or spoof)",
                                     type=["png", "jpg", "jpeg"], key="al_face")
        if face_file is None:
            st.info("Upload a face image, or try a sample spoof from data/spoof/.")
            return
        img = Image.open(face_file)
        st.image(img, caption="Input", width=220)

        from modules.attacks_liveness import (
            minifasnet_liveness, heuristic_liveness)
        from utils.face import detect_faces
        boxes = detect_faces(img)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("MiniFASNet (deep)")
            try:
                r = minifasnet_liveness(img, crop_box=boxes[0] if boxes else None)
                if r["status"] == "live":
                    st.success(f"LIVE - {r['predicted_class']} (live_score={r['live_score']})")
                else:
                    st.error(f"SPOOF - {r['predicted_class']} (print={r['print_score']}, replay={r['replay_score']})")
                st.json(r)
            except FileNotFoundError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"MiniFASNet failed: {e}")
        with c2:
            st.subheader("Heuristic (baseline)")
            r2 = heuristic_liveness(img)
            if r2["status"] == "live":
                st.success(f"LIVE (score {r2['confidence']}%)")
            else:
                st.error(f"SPOOF (score {r2['confidence']}%)")
            st.json(r2)

    with tab_attack:
        from modules.attacks_liveness import attack_taxonomy
        st.markdown("### Attack taxonomy and defences")
        st.dataframe(attack_taxonomy(), use_container_width=True)

    with tab_table:
        from modules.attacks_liveness import defence_comparison_table
        st.markdown("### Liveness / anti-spoofing methods compared")
        st.dataframe(defence_comparison_table(), use_container_width=True)
        st.markdown(
            """
            <div class="premium-info-card">
                <h4>📚 Survey Task (Student 4)</h4>
                <p>Expand to 15+ papers. <em>Seed refs:</em> Silent-Face-Anti-Spoofing/MiniFASNet (Yu 2020), CASIA-FASD / Replay-Attack datasets, Deepfake detection (Rossler 2019 FaceForensics++), adversarial attacks on face recognition (Dong 2019), rPPG liveness (Poh 2010), presentation attack taxonomy (ISO 30107).</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = sidebar()
    if page.startswith("Home"):
        page_home()
    elif page.startswith("Feature"):
        page_feature_extraction()
    elif page.startswith("Template"):
        page_template_protection()
    elif page.startswith("Deep"):
        page_deep_models()
    elif page.startswith("Attacks"):
        page_attacks_liveness()


if __name__ == "__main__":
    main()

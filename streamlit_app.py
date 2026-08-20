from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retinopathy.predict import OrdinalRetinopathyPredictor  # noqa: E402
from retinopathy.quality import assess_image_quality  # noqa: E402

MODEL_PATH = Path("models/retinopathy_ordinal_384.pt")


@st.cache_resource(show_spinner="Loading the research model…")
def load_predictor() -> OrdinalRetinopathyPredictor:
    if not MODEL_PATH.exists():
        raise RuntimeError("The trained model artifact is unavailable in this deployment.")
    return OrdinalRetinopathyPredictor(str(MODEL_PATH), device="cpu")


def show_research_findings() -> None:
    st.title("Research findings")
    st.error(
        "Research use only. This study is not clinically validated and must not be used "
        "for diagnosis, treatment, or decisions about medical care."
    )
    st.write(
        "This project studies five-level diabetic-retinopathy grading from retinal fundus "
        "photographs. The PyTorch model uses an EfficientNet-B0 backbone with four cumulative "
        "ordinal outputs, then groups grades 2 through 4 as referable diabetic retinopathy."
    )

    st.header("Evaluation")
    st.caption(
        "The final checkpoint was evaluated on held-out APTOS data, the official IDRiD test "
        "set, and DeepDRiD's official online evaluation split."
    )
    st.table(
        [
            {
                "Metric": "Quadratic weighted kappa",
                "APTOS test": "0.881 (0.849–0.909)",
                "IDRiD official test": "0.731 (0.590–0.847)",
                "DeepDRiD official evaluation": "0.612 (0.499–0.702)",
            },
            {
                "Metric": "Referable-DR AUROC",
                "APTOS test": "0.981",
                "IDRiD official test": "0.931",
                "DeepDRiD official evaluation": "0.905",
            },
            {
                "Metric": "Referable-DR sensitivity",
                "APTOS test": "93.6%",
                "IDRiD official test": "85.9%",
                "DeepDRiD official evaluation": "55.5%",
            },
            {
                "Metric": "Referable-DR specificity",
                "APTOS test": "92.6%",
                "IDRiD official test": "87.2%",
                "DeepDRiD official evaluation": "96.6%",
            },
            {
                "Metric": "Expected calibration error",
                "APTOS test": "2.5%",
                "IDRiD official test": "13.5%",
                "DeepDRiD official evaluation": "23.1%",
            },
        ]
    )

    st.header("Data and study controls")
    st.markdown(
        """
        - Audited 3,662 APTOS-derived images before splitting. The audit found 251 duplicate
          rows and excluded 30 hashes with conflicting labels, leaving 3,504 unique,
          non-conflicting images.
        - Linked 3,201 high-resolution images to cleaned APTOS records with mutual-nearest
          retinal perceptual-hash matching and independent binary-label agreement.
        - Fine-tuned for three epochs with IDRiD's official training split. The IDRiD official
          test split remained outside training, checkpoint selection, and calibration.
        - Ran DeepDRiD evaluation after freezing the final checkpoint, without adjusting the
          threshold or model.
        """
    )

    st.header("What the results mean")
    st.write(
        "Fine-tuning raised severe-grade recall on IDRiD from 10.5% (2/19) to 84.2% (16/19), "
        "while APTOS severe recall fell from 37.0% to 29.6%. DeepDRiD offered a harder external "
        "check: specificity remained high, but sensitivity, minority-grade recall, and "
        "calibration declined. These cross-dataset differences are the central result of the "
        "study, rather than a clinical performance claim."
    )

    st.header("Research limitations")
    st.markdown(
        """
        - The datasets are limited in size and class balance, and the APTOS-derived split is
          image-level because patient identifiers were unavailable.
        - DeepDRiD referable sensitivity was 55.5% despite 96.6% specificity. The model is not
          reliable as a screening system across acquisition settings.
        - Image-quality checks and Grad-CAM overlays are engineering aids. They are not clinical
          validation or medical reasoning.
        - The study has not been evaluated prospectively or inside a clinical workflow.
        """
    )
    st.link_button(
        "Read the full methods and model card on GitHub",
        "https://github.com/shlokbhutani13/retinopathy-grading",
    )


def main() -> None:
    st.set_page_config(page_title="Retinopathy Grading", page_icon="👁️", layout="centered")
    page = st.sidebar.radio("Explore", ["Prediction demo", "Research findings"])
    if page == "Research findings":
        show_research_findings()
        return

    st.title("Retinal Image Classification for Diabetic Retinopathy")
    st.error(
        "Research use only. This educational model is not clinically validated and must not "
        "be used for diagnosis, treatment, or decisions about medical care."
    )
    st.caption(
        "Five ordered grades • PyTorch EfficientNet-B0 • APTOS holdout AUROC 0.981 for "
        "referable diabetic retinopathy"
    )

    upload = st.file_uploader(
        "Upload a retinal fundus photograph", type=["jpg", "jpeg", "png"],
        help=(
            "Use only images you are authorized to upload. "
            "Do not upload personally identifying data."
        ),
    )
    if upload is None:
        st.info("The model checks image quality before generating a research prediction.")
        return

    image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not st.button("Run research prediction", type="primary"):
        return

    quality = assess_image_quality(image)
    if not quality["acceptable"]:
        st.warning("Image quality is unsuitable: " + ", ".join(quality["reasons"]))
        return

    with st.spinner("Analyzing image…"):
        result, overlay = load_predictor().predict(image)

    st.subheader(f"Predicted grade: {result['grade_name']}")
    st.write(
        "This image is marked for professional examination."
        if result["referable_dr"]
        else "This image is not marked as referable by this research model."
    )
    st.metric("Model confidence", f"{float(result['confidence']):.0%}")
    if result["low_confidence"]:
        st.warning("Low model confidence: treat this output as especially uncertain.")
    st.bar_chart(result["probabilities"])
    st.image(
        overlay,
        caption="Grad-CAM attention overlay (research explanation only)",
        use_container_width=True,
    )
    st.caption("Image quality checks passed. This is not a medical diagnosis.")


if __name__ == "__main__":
    main()

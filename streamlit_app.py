from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from retinopathy.predict import OrdinalRetinopathyPredictor
from retinopathy.quality import assess_image_quality

MODEL_PATH = Path("models/retinopathy_ordinal_384.pt")


@st.cache_resource(show_spinner="Loading the research model…")
def load_predictor() -> OrdinalRetinopathyPredictor:
    if not MODEL_PATH.exists():
        raise RuntimeError("The trained model artifact is unavailable in this deployment.")
    return OrdinalRetinopathyPredictor(str(MODEL_PATH), device="cpu")


def main() -> None:
    st.set_page_config(page_title="Retinopathy Grading", page_icon="👁️", layout="centered")
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

from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from retinopathy.predict import OrdinalRetinopathyPredictor, RetinopathyPredictor
from retinopathy.quality import assess_image_quality

MODEL_PATH = Path("models/retinopathy_efficientnet_b0.pt")
ORDINAL_MODEL_PATH = Path("models/retinopathy_ordinal_384.pt")


def analyze_image(
    image: Image.Image | None,
    predictor: RetinopathyPredictor,
) -> tuple[str, dict[str, float], Image.Image]:
    if image is None:
        raise ValueError("Upload a retinal fundus image before running the model.")
    quality = assess_image_quality(image)
    if not quality["acceptable"]:
        reasons = ", ".join(quality["reasons"])
        raise ValueError(
            "Image quality is unsuitable for grading. Check the retinal field, lighting, "
            f"and focus. Failed checks: {reasons}."
        )

    result, overlay = predictor.predict(image)
    screening = (
        "The model marks this image for professional examination."
        if result["referable_dr"]
        else "The model does not mark this image as referable diabetic retinopathy."
    )
    uncertainty = (
        " The confidence is low, so this result is especially uncertain."
        if result["low_confidence"]
        else ""
    )
    summary = (
        f"### Predicted grade: {result['grade_name']}\n\n"
        f"{screening}{uncertainty}\n\n"
        f"Model confidence: **{float(result['confidence']):.0%}**\n\n"
        "Image quality checks: **passed**\n\n"
        "**Research use only:** this output is not a diagnosis and cannot replace an eye "
        "examination by a qualified professional."
    )
    return summary, result["probabilities"], overlay


def build_demo(predictor: RetinopathyPredictor | None = None) -> gr.Blocks:
    with gr.Blocks(title="Retinopathy Grading") as demo:
        gr.Markdown(
            """
            # Retinopathy Grading
            Upload a retinal fundus photograph to explore a five-grade research model and
            its visual explanation.

            <div class="research-note">
            This educational model is not clinically validated and must not be used for
            diagnosis, treatment, or decisions about medical care.
            </div>
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                    label="Retinal fundus image",
                )
                analyze = gr.Button("Analyze image", variant="primary")
            with gr.Column():
                summary = gr.Markdown("Upload an image to begin.")
                probabilities = gr.Label(label="Grade probabilities", num_top_classes=5)
        explanation = gr.Image(label="Model attention overlay", interactive=False)

        def run(image):
            if predictor is None:
                raise gr.Error(
                    "The trained model artifact is not available. Follow the training "
                    "instructions in the repository."
                )
            return analyze_image(image, predictor)

        analyze.click(
            run,
            inputs=image_input,
            outputs=[summary, probabilities, explanation],
        )
    return demo


if __name__ == "__main__":
    if ORDINAL_MODEL_PATH.exists():
        loaded_predictor = OrdinalRetinopathyPredictor(str(ORDINAL_MODEL_PATH))
    elif MODEL_PATH.exists():
        loaded_predictor = RetinopathyPredictor(str(MODEL_PATH))
    else:
        loaded_predictor = None
    build_demo(loaded_predictor).launch()

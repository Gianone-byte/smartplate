"""
SmartPlate Gradio application — inference only, no training code here.

Run locally:
    python app.py

On Hugging Face Spaces, this file is loaded automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

# Monkey-patch for Gradio 4.36 bool-iteration bug
import gradio_client.utils as _gradio_client_utils
_original_get_type = _gradio_client_utils.get_type

def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _original_get_type(schema)

_gradio_client_utils.get_type = _patched_get_type

from PIL import Image

from src.pipeline import SmartPlatePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pipeline: Optional[SmartPlatePipeline] = None


def get_pipeline() -> SmartPlatePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SmartPlatePipeline()
    return _pipeline


def analyze_meal(
    image: Optional[Image.Image],
    user_question: str,
) -> Tuple[str, str, str]:
    """Gradio callback: run the full pipeline and return formatted outputs.

    Returns:
        Tuple of (cv_output, ml_output, nlp_output) as Markdown strings.
    """
    if image is None:
        return "Please upload a meal photo to get started.", "", ""

    question = user_question.strip() if user_question else None

    try:
        result = get_pipeline().process(image, user_question=question)

        cv = result["cv_result"]
        ml = result["ml_result"]
        nlp = result["nlp_result"]

        # --- CV output ---
        food_name = cv["class"].replace("_", " ").title()
        cv_text = f"**{food_name}**\n\nConfidence: {cv['confidence']:.0%}"
        if cv.get("top_5") and len(cv["top_5"]) > 1:
            top5_lines = "\n".join(
                f"- {r['class'].replace('_', ' ').title()}: {r['confidence']:.0%}"
                for r in cv["top_5"]
            )
            cv_text += f"\n\n**Top 5 predictions:**\n{top5_lines}"

        # --- ML output ---
        n = ml["nutrition"]
        health = ml["health_label"].upper()
        health_emoji = {"HEALTHY": "🟢", "MEDIUM": "🟡", "UNHEALTHY": "🔴"}.get(
            health, "⚪"
        )
        proba = ml.get("probabilities", {})
        proba_str = "  ".join(
            f"{k}: {v:.0%}" for k, v in proba.items()
        )

        ml_text = (
            f"### Nutritional Values (per 100 g)\n\n"
            f"| Nutrient | Amount |\n"
            f"|---|---|\n"
            f"| Energy | {n['kcal']:.0f} kcal |\n"
            f"| Fat | {n['fat']:.1f} g |\n"
            f"| — Saturated fat | {n['sat_fat']:.1f} g |\n"
            f"| Carbohydrates | {n['carbs']:.1f} g |\n"
            f"| — Sugars | {n['sugar']:.1f} g |\n"
            f"| Fiber | {n['fiber']:.1f} g |\n"
            f"| Protein | {n['protein']:.1f} g |\n"
            f"| Salt | {n['salt']:.1f} g |\n\n"
            f"**Health Category:** {health_emoji} {health}\n\n"
            f"*Confidence: {proba_str}*"
        )

        # --- NLP output ---
        sources = nlp.get("sources", [])
        sources_str = " · ".join(dict.fromkeys(sources)) if sources else "WHO · DGE · Harvard"
        nlp_text = f"{nlp['answer']}\n\n*Sources: {sources_str}*"

        return cv_text, ml_text, nlp_text

    except EnvironmentError as exc:
        logger.error("Environment error: %s", exc)
        return (
            "⚠️ Configuration error.",
            str(exc),
            "Please set OPENAI_API_KEY in your .env file.",
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        return f"⚠️ Error: {exc}", "", "Please try again or check the logs."


# ── Gradio layout ──────────────────────────────────────────────────────────────

_EXAMPLES_DIR = Path("assets/examples")


def _find_examples() -> list:
    """Return example image paths if the directory exists."""
    if not _EXAMPLES_DIR.exists():
        return []
    paths = sorted(
        list(_EXAMPLES_DIR.glob("*.jpg"))
        + list(_EXAMPLES_DIR.glob("*.jpeg"))
        + list(_EXAMPLES_DIR.glob("*.png"))
    )
    return [[str(p), ""] for p in paths[:5]]


with gr.Blocks(
    title="SmartPlate – AI Nutrition Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# SmartPlate – AI Nutrition Assistant 🍽️\n"
        "Photograph your meal and get instant nutritional analysis with "
        "evidence-based health advice."
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload a meal photo")
            question_input = gr.Textbox(
                label="Ask a question (optional)",
                placeholder="e.g. Can I eat this on a diet?",
                lines=2,
            )
            submit_btn = gr.Button("Analyze 🔍", variant="primary")

        with gr.Column(scale=2):
            cv_output = gr.Markdown(label="Dish Recognition")
            ml_output = gr.Markdown(label="Nutritional Analysis")
            nlp_output = gr.Markdown(label="Health Advice")

    examples = _find_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[img_input, question_input],
            label="Try an example",
        )

    submit_btn.click(
        fn=analyze_meal,
        inputs=[img_input, question_input],
        outputs=[cv_output, ml_output, nlp_output],
    )

    gr.Markdown(
        "---\n"
        "**Sources:** WHO · DGE (Deutsche Gesellschaft für Ernährung) · Harvard Nutrition\n\n"
        "*For educational use only — not medical advice. "
        "ZHAW KI-Anwendungen FS 2026.*"
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(show_api=False)
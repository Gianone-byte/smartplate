"""
SmartPlate Gradio application — inference only, no training code here.

Run locally:
    python app.py

On Hugging Face Spaces, this file is loaded automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from src.pipeline import SmartPlatePipeline, PipelineResult

load_dotenv()

# Lazy-loaded pipeline (initialized on first request to avoid cold-start issues)
_pipeline: SmartPlatePipeline | None = None


def get_pipeline() -> SmartPlatePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SmartPlatePipeline()
    return _pipeline


def analyze_meal(image: Image.Image) -> tuple[str, str, str]:
    """Gradio callback: run the full pipeline and return formatted outputs.

    Args:
        image: PIL Image uploaded by the user.

    Returns:
        A tuple of three strings for the three Gradio output components:
        - ``dish_info``: Dish label + confidence
        - ``nutrition_info``: Nutritional values + health score
        - ``advice``: LLM-generated explanation + tips
    """
    if image is None:
        return "No image provided.", "", ""

    try:
        result: PipelineResult = get_pipeline().run(image)

        dish_info = (
            f"**{result.food_label.replace('_', ' ').title()}**\n"
            f"Confidence: {result.cv_confidence:.0%}"
        )

        n = result.nutrition
        nutrition_info = (
            f"### Nutritional Values (per 100g)\n"
            f"| Nutrient | Amount |\n"
            f"|---|---|\n"
            f"| Energy | {n.energy_kcal:.0f} kcal |\n"
            f"| Fat | {n.fat_g:.1f} g |\n"
            f"| — Saturated fat | {n.saturated_fat_g:.1f} g |\n"
            f"| Sugars | {n.sugars_g:.1f} g |\n"
            f"| Fiber | {n.fiber_g:.1f} g |\n"
            f"| Protein | {n.proteins_g:.1f} g |\n"
            f"| Salt | {n.salt_g:.1f} g |\n\n"
            f"**Health Score:** {n.health_score:.0f}/100 — {n.health_label.upper()}"
        )

        advice = (
            f"### Analysis\n{result.rag.explanation}\n\n"
            f"### Tips\n{result.rag.tips}\n\n"
            f"*Sources: {', '.join(result.rag.sources)}*"
        )

        return dish_info, nutrition_info, advice

    except NotImplementedError:
        return (
            "Models not yet trained.",
            "Run the training notebooks first.",
            "See README.md for instructions.",
        )
    except Exception as e:
        return f"Error: {e}", "", ""


with gr.Blocks(title="SmartPlate", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SmartPlate\nPhotograph your meal — get instant nutritional analysis and personalized advice.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload meal photo")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            dish_output = gr.Markdown(label="Dish")
            nutrition_output = gr.Markdown(label="Nutrition")
            advice_output = gr.Markdown(label="Health Advice")

    analyze_btn.click(
        fn=analyze_meal,
        inputs=[image_input],
        outputs=[dish_output, nutrition_output, advice_output],
    )

    gr.Markdown("---\n*ZHAW KI-Anwendungen FS 2026 | For educational use only — not medical advice.*")

if __name__ == "__main__":
    demo.launch()

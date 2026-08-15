import os
import sys

# Make sure the project root (parent of this app/ folder) is on the path,
# so `from src.inference import ...` works even when running `python app/app.py` directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.inference import SentimentAnalyzer

analyzer = SentimentAnalyzer()  # reads SENTIMENT_BACKEND env var, defaults to "classic"


def analyze(text):
    result = analyzer.predict(text)
    if result["label"] is None:
        return "Enter some text first.", {}
    label = result["label"]
    scores = result["scores"]
    verdict = f"{'Positive' if label == 'positive' else 'Negative'} ({scores.get(label, 0) * 100:.1f}% confident)"
    return verdict, scores


with gr.Blocks(title="Sentimeter") as demo:
    gr.Markdown("## sentimeter\nType a sentence or word and see the predicted sentiment.")

    with gr.Row():
        text_input = gr.Textbox(
            label="Your text",
            placeholder="e.g. The plot was predictable but the acting really saved the film.",
            lines=3,
        )

    analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        verdict_output = gr.Textbox(label="Prediction", interactive=False)

    scores_output = gr.Label(label="Sentiment breakdown", num_top_classes=2)

    gr.Examples(
        examples=[
            "Absolute waste of time, I want those two hours back.",
            "One of the best things I've watched this year.",
            "It was okay, nothing special but not bad either.",
        ],
        inputs=text_input,
    )

    analyze_btn.click(fn=analyze, inputs=text_input, outputs=[verdict_output, scores_output])
    text_input.submit(fn=analyze, inputs=text_input, outputs=[verdict_output, scores_output])

if __name__ == "__main__":
    demo.launch()

import gradio as gr
from agent import study_helper, evaluate_summary

def process_notes(notes):
    output = study_helper(notes)
    score = evaluate_summary(notes, output)
    return output, f"Coverage Score: {score}%"

gr.Interface(
    fn=process_notes,
    inputs=gr.Textbox(lines=6, label="Enter your study notes here"),
    outputs=[gr.Textbox(label="AI Summary + Quiz"), gr.Textbox(label="Evaluation Score")],
    title="Smart Study Assistant"
).launch()

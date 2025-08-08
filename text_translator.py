import torch
import gradio as gr
import json
from transformers import pipeline
import pyttsx3
import tempfile
import os

# Load the NLLB model
model_path = "../text_translator/model/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
translator = pipeline("translation", model=model_path)

# translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
# Load language codes
def load_lang_codes(file_path='../text_translator/lang file/lang_code.json'):
# def load_lang_codes(file_path='lang_code.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {entry['Language']: entry['FLORES-200 code'] for entry in data}
    except Exception as e:
        print(f"Error loading language codes: {e}")
        return {}

lang_dict = load_lang_codes()
language_list = list(lang_dict.keys())

# Text-to-Speech function using pyttsx3
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)

    # Save speech to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        filename = fp.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# Translation + TTS function
def translate_and_speak(text, src_lang, tgt_lang):
    src_code = lang_dict.get(src_lang)
    tgt_code = lang_dict.get(tgt_lang)

    if not src_code or not tgt_code:
        return "Invalid language selected.", None

    try:
        result = translator(text, src_lang=src_code, tgt_lang=tgt_code)
        translated_text = result[0]['translation_text']
        audio_path = text_to_speech(translated_text)
        return translated_text, audio_path
    except Exception as e:
        return f"Translation error: {e}", None

# Gradio UI
with gr.Blocks(css="""
    .gradio-container {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f4f4;
        padding: 40px;
    }
    h1 {
        color: black !important;
        font-size: 28px !important;
        margin-bottom: 30px !important;
    }
    .gr-button {
        font-size: 16px !important;
        padding: 10px 20px !important;
    }
    .gr-box {
        padding: 20px !important;
    }
""") as demo:
    gr.Markdown("<h1 style='text-align: center;'>@GenAI MultiLanguage Translator Model</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="🌐 Enter Text",
                placeholder="Type something to translate...",
                lines=4
            )
            gr.Markdown("<br>")
            src_dropdown = gr.Dropdown(choices=language_list, label="📤 Source Language", value="English")
            tgt_dropdown = gr.Dropdown(choices=language_list, label="📥 Target Language", value="French")
            gr.Markdown("<br>")
            translate_button = gr.Button("🔄 Translate", variant="primary")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="✅ Translated Text", lines=6, interactive=False)
            audio_output = gr.Audio(label="🔊 Speak Translation", type="filepath", autoplay=True)

    translate_button.click(fn=translate_and_speak,
                           inputs=[input_text, src_dropdown, tgt_dropdown],
                           outputs=[output_text, audio_output])

# Launch
demo.launch()

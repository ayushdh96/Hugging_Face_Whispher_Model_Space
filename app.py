import torch
from transformers import pipeline
import gradio as gr

# Setup device 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the ASR model pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",
    chunk_length_s=30,
    device=device,
)

# Function to make prediction from audio input
def transcribe(audio):
    # Convert Gradio input to the format expected by the ASR pipeline
    prediction = pipe(audio, batch_size=8)["text"]
    return prediction

# Define the Gradio interface
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),  # Removed 'source' argument
    outputs="text",
    title="Speech to Text with Whisper Model",
    description="Record your voice and transcribe it to text using OpenAI Whisper model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)

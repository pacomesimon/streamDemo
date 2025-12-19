import os
import gradio as gr
from ollama import chat
from ollama import ChatResponse
import requests
from PIL import Image

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class AudioToText:
    def __init__(self):
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
      torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

      model_id = "openai/whisper-tiny"

      model = AutoModelForSpeechSeq2Seq.from_pretrained(
          model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
      )
      model.to(device)

      processor = AutoProcessor.from_pretrained(model_id)

      self.pipe = pipeline(
          "automatic-speech-recognition",
          model=model,
          tokenizer=processor.tokenizer,
          feature_extractor=processor.feature_extractor,
          torch_dtype=torch_dtype,
          device=device,
      )
    def transcribe(self, filename):
        result = self.pipe(filename, batch_size=16,return_timestamps=True)["text"]
        return result
audio_pipe = AudioToText()


def get_audio_summary(audio_file = None, system_prompt = None):
  if not audio_file:
    return 'no audio file provided'
  transcription = audio_pipe.transcribe(audio_file)
  if (system_prompt is None) or (len(str(system_prompt).strip())==0):
    for _ in [transcription]:
      yield _
  else:
    system_message_dict = {
        "role": "system", 
        "content": system_prompt}
    user_message_dict = {"role": "user", "content": f"transcription: {transcription}"}
    messages = [system_message_dict, user_message_dict]
    result = ""
    for chunk in chat(model='amsaravi/medgemma-4b-it:q8', messages=messages, stream = True):
      result += chunk['message']['content']
      yield result


def chat_with_ollama(message, history, image, system_prompt = None):
    # print(f"message: {message}")
    # print(f"history: {history}")
    messages = []
    # # Add previous conversation history
    # for human, assistant in history:
    #     messages.append({"role": "user", "content": human})
    #     messages.append({"role": "assistant", "content": assistant})

    if (system_prompt is not None) and (len(system_prompt.strip())>0):
        system_message_dict = {
            "role": "system", 
            "content": system_prompt}
        messages.append(system_message_dict)

    # Add current user message
    user_message_dict = {"role": "user", "content": message}
    if image:
          from io import BytesIO
          buffered = BytesIO()
          image.save(buffered, format="PNG") # Assuming PNG format is suitable for base64 encoding
          img_b64 = base64.b64encode(buffered.getvalue()).decode()
          user_message_dict["images"] = [img_b64]

    messages.append(user_message_dict)
    result = ""
    for chunk in chat(model='amsaravi/medgemma-4b-it:q8', messages=messages, stream = True):
      result += chunk['message']['content']
      yield result


with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column():
      ollama_server_btn = gr.Button("Restart Ollama Server")
      ollama_server_btn.click(
        fn=lambda : os.system("sudo ollama serve"), 
        inputs = [], outputs = [])
    with gr.Column():
      ollama_model_btn = gr.Button("Reload Ollama Model")
      ollama_model_btn.click(
        fn=lambda : os.system("sudo ollama pull amsaravi/medgemma-4b-it:q8"),
        inputs = [], outputs = [])

  with gr.Tab("Chat with MedGemma"):
    gr.ChatInterface(
        fn=chat_with_ollama,
        title="Ollama Chat with MedGemma (Multimodal)",
        additional_inputs=[gr.Image(type="pil", label="Upload Image",height="20vh", render=False),
                           gr.TextArea(label="System Prompt")
                          
                           ],
        examples=[["What abnormalities do you see in this X-ray?", 
                   "./chest_xray.png",
                   "you are a medical assistant"],
                  ],
    )
  with gr.Tab("Audio to Text"):
    gr.Interface(
        fn=get_audio_summary,
        inputs = [gr.Audio(sources=['upload', 'microphone'],
                      type="filepath", label="Upload Audio", render=False),
                  gr.TextArea(label="System Prompt")
                  ],
        outputs = gr.TextArea(label="Output Text")
    )
demo.launch(
    # debug=True,
    share=True,
)
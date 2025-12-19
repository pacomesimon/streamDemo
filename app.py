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

def restart_ollama_server():
  log_file = "output1.log"
  os.system(f"nohup ollama serve > {log_file} 2>&1 &")
  while True:
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            yield f.read()

def reload_ollama_model():
  log_file = "output2.log"
  os.system(f"ollama pull amsaravi/medgemma-4b-it:q8 > {log_file} 2>&1 &")
  while True:
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            yield f.read()

with gr.Blocks() as demo:
  with gr.Tab("Ollama Server Control"):
    with gr.Column():
      ollama_server_btn = gr.Button("Restart Ollama Server")
      server_logs = gr.Textbox(label="Ollama Server Logs", lines=10)
      ollama_server_btn.click(
        fn=restart_ollama_server,
        inputs = [], outputs = server_logs)
    with gr.Column():
      ollama_model_btn = gr.Button("Reload Ollama Model")
      model_logs = gr.Textbox(label="Ollama Model Reload Logs", lines=10)
      ollama_model_btn.click(
        fn=reload_ollama_model,
        inputs = [], outputs = model_logs)

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
  with gr.Tab("JS Documentation"):
    gr.Markdown(
      """
      ## First, install the needed package:
      ```bash
      npm i -D @gradio/client
      ```
      """
    )
    gr.Markdown(
        """
        **Vision prompting example:**
          Remarks: 
          - Make sure to replace the URL with your Gradio app URL.
          - You can pass the imageBlob for visual analysis.
          - The system prompt is optional.
          - The response is streamed.

          ```javascript
          import { Client } from "@gradio/client";
          import fs from "fs/promises";
            const buffer = await fs.readFile("./chest_xray.png");
            const URL = "https://9c1cc7f2462b816796.gradio.live"; // Replace with your Gradio app URL
            const imageBlob = new Blob([buffer], { type: "image/png" });
            const client = await Client.connect(URL);
            let result = client.submit("/chat", {
                message: "Who are you?",
                    image: null, // You can pass the imageBlob for visual analysis.
                system_prompt: "You are a clinical analysis system",
            });
            let current_message = ""
            for await (const message of result) {
            console.clear();
            current_message = message["data"][0]
            console.log(current_message);
            }
          ```
        """

    )
    gr.Markdown(
        """
        **Audio prompting example**
        Remarks:
        - Make sure to replace the URL with your Gradio app URL.
        - You can pass the audioBlob for transcription.
        - Without the system prompt, the model will just transcribe the audio.
        - With the system prompt, the model will provide a summary or analysis based on the transcription.

        ```javascript
        import { Client } from "@gradio/client";
        import fs from "fs/promises";
          const URL = "https://9c1cc7f2462b816796.gradio.live/"; // Replace with your Gradio app URL
          const buffer = await fs.readFile("./hello_there.mp3");
          const audioBlob = new Blob([buffer], { type: "audio/mp3" });
          const client = await Client.connect(URL);
          let result = client.submit("/predict", {
                  audio_file: audioBlob,
              system_prompt: "This is an audio transcription. what do you think it's all about?",
          });

          let current_message = ""
          for await (const message of result) {
          console.clear();
          current_message = message["data"][0]
          console.log(current_message);
          }

        ```
        """
    )
demo.launch(
    # debug=True,
    share=True,
)
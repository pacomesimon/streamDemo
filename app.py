import os
import gradio as gr
from ollama import chat
from ollama import ChatResponse
import requests
from PIL import Image
import base64
import json
from io import BytesIO
import tempfile
import mimetypes

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class AudioToText:
    def __init__(self):
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
      torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

      model_id = "openai/whisper-tiny"

      model = AutoModelForSpeechSeq2Seq.from_pretrained(
          model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
      )
      model.to(device)

      processor = AutoProcessor.from_pretrained(model_id)

      self.pipe = pipeline(
          "automatic-speech-recognition",
          model=model,
          tokenizer=processor.tokenizer,
          feature_extractor=processor.feature_extractor,
          dtype=torch_dtype,
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

def get_b64_string_from_url(url):
  header, b64_data = url.split(",", 1)
  return b64_data

def data_uri_to_file(data_uri):
    header, b64_data = data_uri.split(",", 1)

    # Example header: data:audio/mpeg;base64
    assert ";base64" in header, "Only base64 data URIs are supported"

    mime_type = header.split(":")[1].split(";")[0]
    extension = mimetypes.guess_extension(mime_type) or ""

    data_bytes = base64.b64decode(b64_data)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    tmp.write(data_bytes)
    tmp.flush()
    tmp.close()

    return tmp.name, mime_type

def chat_with_ollama_JSON(messages):
    messages = json.loads(messages)
    system_message_audio_transcript = None
    for mdx, message in enumerate(messages):
      text_content,images = None,[]
      for content_item in message["content"]:
        if content_item["type"] == "text":
          text_content = content_item["text"]
        if content_item["type"] == "image_url":
          image_b64 = get_b64_string_from_url(
            content_item["image_url"]['url']
            )
          images.append(image_b64)
        if content_item['type'] == "audio_url":
          audio_temp_filepath, _ = data_uri_to_file(content_item["audio_url"]['url'])
          audio_transcription = audio_pipe.transcribe(audio_temp_filepath)
          system_message_audio_transcript = {
            "role": "system", 
            "content": f"The user has attached an audio with this transcription:\n {audio_transcription}"
          }
      removed_content_key = messages[mdx].pop('content', 'Content Key Not Found')
      if not(text_content is None):
        messages[mdx]["content"] = text_content
      if not(len(images)==0):
        messages[mdx]["images"] = images
    
    if not(system_message_audio_transcript is None):
      messages.append(system_message_audio_transcript)

    # result = ""
    result = []
    for chunk in chat(model='amsaravi/medgemma-4b-it:q8', messages=messages, stream = True):
      # result += chunk['message']['content']
      chunk_dict = dict(chunk)
      chunk_dict["message"] = dict(chunk["message"])
      result.append(chunk_dict)
      yield json.dumps(result, sort_keys=False, indent=4)

def mp3_to_b64(mp3_path):
    with open(mp3_path, "rb") as f:
        mp3_bytes = f.read()
    mp3_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
    return mp3_b64

def get_b64(image):
  buffered = BytesIO()
  image.save(buffered, format="PNG") # Assuming PNG format is suitable for base64 encoding
  img_b64 = base64.b64encode(buffered.getvalue()).decode()
  return img_b64

def assemble_json_prompt(system_prompt_txt, 
      user_prompt_txt, img_b64_txt, audio_b64_txt):
  messages = []
  if (system_prompt_txt is not None) and (len(str(system_prompt_txt).strip())!=0):
    system_message_dict = {
            "role": "system", 
            "content": [
              {
                "type":"text",
                "text":system_prompt_txt
              }
            ]
            }
    messages.append(system_message_dict)
  if (user_prompt_txt is not None) and (len(str(user_prompt_txt).strip())!=0):
    user_message_dict = {
            "role": "user", 
            "content": [
              {
                "type":"text",
                "text":user_prompt_txt
              }
            ]
            }
    if (img_b64_txt is not None) and (len(str(img_b64_txt).strip())!=0):
      user_message_dict["content"].append(
        {
          "type":"image_url",
          "image_url":{
            'url': f"data:image/png;base64,{img_b64_txt}"
          }
        }
      )
      # user_message_dict["images"] = [img_b64_txt]
    if (audio_b64_txt is not None) and (len(str(audio_b64_txt).strip())!=0):
      user_message_dict["content"].append(
        {
          "type":"audio_url",
          "audio_url":{
            'url': f"data:audio/mpeg;base64,{audio_b64_txt}"
          }
        }
      )
    messages.append(user_message_dict)
  return json.dumps(messages, sort_keys=False, indent=4)

  

def chat_with_ollama(message, history, image, system_prompt = None):
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
  
  with gr.Tab("Chat with MedGemma JSON"):
    with gr.Row():
      with gr.Column():
        input_img = gr.Image(type="pil", label="Upload Image",height="20vh")
        examples_img = gr.Examples(
          examples=["./chest_xray.png"],
          inputs=[input_img],
        )
        img_b64_txt = gr.Textbox(label="Image base64")
        input_img.change(
          fn = get_b64,
          inputs = [input_img],
          outputs = [img_b64_txt]
        )
      with gr.Column():
        input_audio = gr.Audio(sources=['upload', 'microphone'],
                      type="filepath", label="Upload Audio")
        examples_audio = gr.Examples(
          examples=["./hello_there.mp3"],
          inputs=[input_audio],
        )
        audio_b64_txt = gr.Textbox(label="Audio base64")
        input_audio.change(
          fn = mp3_to_b64,
          inputs = [input_audio],
          outputs = [audio_b64_txt]
        )
    with gr.Row():
        system_prompt_txt = gr.Textbox(label="System Prompt")
        user_prompt_txt = gr.Textbox(label="User Prompt")
    with gr.Row():
      assemble_prompt_btn = gr.Button("Assemble JSON")
      prompt_JSON_txt = gr.TextArea(label="JSON Prompt")
      assemble_prompt_btn.click(
        fn = assemble_json_prompt,
        inputs = (system_prompt_txt, user_prompt_txt, img_b64_txt, audio_b64_txt),
        outputs = [prompt_JSON_txt]
      )
    with gr.Row():
      prompt_ollama_btn = gr.Button("send prompt to ollama")
      ollama_output_txt = gr.Markdown()
      prompt_ollama_btn.click(
        fn = chat_with_ollama_JSON,
        inputs = [prompt_JSON_txt],
        outputs = [ollama_output_txt]
      )

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
          ### Remarks: 
          - Make sure to replace the URL with your Gradio app URL.
          - The response is streamed.

          ```javascript
import { Client } from "@gradio/client";
import fs from "fs/promises";

const URL = "http://127.0.0.1:7860"; // Replace with your Gradio app URL
const client = await Client.connect(URL);

const buffer = await fs.readFile("../chest_xray.png");
const result = client.submit("/chat_with_ollama_JSON", { 		
			messages: JSON.stringify([
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "you are a medical assistant."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what's in the picture?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": `data:image/png;base64,${buffer.toString("base64")}`
                }
            }
        ]
    }
]), 
	});

let cummulative_message = ""
let json_data = ""
for await (const response of result) {
  console.clear();
  json_data = response["data"][0]
  json_data = JSON.parse(json_data)
  cummulative_message += json_data.at(-1)["message"]["content"]
  console.log(cummulative_message);
  if(json_data.at(-1)["done_reason"]=="stop"){
    result.cancel();
    break;
  }
}
console.log("DONE!")
          ```
        """

    )
    gr.Markdown(
        """
        **Audio prompting example:**
        ### Remarks:
        - Make sure to replace the URL with your Gradio app URL.

        ```javascript
import { Client } from "@gradio/client";
import fs from "fs/promises";
const URL = "http://127.0.0.1:7860/"; // Replace with your Gradio app URL
const client = await Client.connect(URL);

const buffer = await fs.readFile("../hello_there.mp3");

const result = client.submit("/chat_with_ollama_JSON", {
        messages: JSON.stringify([
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "you are an assistant."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what's in the audio?"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": `data:audio/mpeg;base64,${buffer.toString("base64")}`
                  }
            }
        ]
    }
]),
});
let cummulative_message = ""
let json_data = ""
for await (const response of result) {
  console.clear();
  json_data = response["data"][0]
  json_data = JSON.parse(json_data)
  cummulative_message += json_data.at(-1)["message"]["content"]
  console.log(cummulative_message);
  if(json_data.at(-1)["done_reason"]=="stop"){
    result.cancel();
    break;
  }
}
console.log("DONE!")

        ```
        """
    )
demo.launch(
    share=True,
)
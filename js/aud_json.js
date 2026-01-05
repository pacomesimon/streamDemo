import { Client } from "@gradio/client";
import fs from "fs/promises";
const URL = "http://127.0.0.1:7860/"; // Replace with your Gradio app URL
const client = await Client.connect(URL);

const buffer = await fs.readFile("../hello_there.mp3");

const result = client.submit("/chat_with_ollama_JSON", {
        messages: JSON.stringify([
    {
        "role": "system",
        "content": "you are a funny assistant. you reply with emojis."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "tell me in summary what the audio is all about."
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
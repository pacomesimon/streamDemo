import { Client } from "@gradio/client";
import fs from "fs/promises";
const URL = "https://0f07c8280493e7c621.gradio.live/"; // Replace with your Gradio app URL
const buffer = await fs.readFile("../hello_there.mp3");
const audioBlob = new Blob([buffer], { type: "audio/mp3" });
const client = await Client.connect(URL);
const result = client.submit("/predict", {
        audio_file: audioBlob,
    system_prompt: "This is an audio transcription. what do you think it's all about?",
});
let previous_message = ""
let current_message = ""
for await (const message of result) {
  previous_message = current_message
  console.clear();
  current_message = message["data"][0]
  console.log(current_message);
  if(previous_message.length == current_message.length){
    result.cancel();
    break;
  }
}
console.log("DONE!")
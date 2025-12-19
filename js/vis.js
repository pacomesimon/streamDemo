import { Client } from "@gradio/client";
import fs from "fs/promises";
const buffer = await fs.readFile("../chest_xray.png");
const URL = "https://22c15a1574ad57ad36.gradio.live/"; // Replace with your Gradio app URL
const imageBlob = new Blob([buffer], { type: "image/png" });
const client = await Client.connect(URL);
const result = client.submit("/chat", {
    message: "Who are you?",
        image: null, // You can pass the imageBlob for visual analysis.
    system_prompt: "You are a clinical analysis system",
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
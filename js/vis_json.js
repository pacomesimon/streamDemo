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
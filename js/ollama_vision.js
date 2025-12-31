import { readFile } from 'node:fs/promises';

async function streamOllama() {
  const imagePath = './streamDemo/chest_xray.png'
  const base64Image = await readFile(imagePath, 'base64');
  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'amsaravi/medgemma-4b-it:q8',
      messages: [{ 
        role: 'user', 
        content: 'In less than 100 words, describe the image.',
        images: [base64Image] // Array of base64 strings
        }],
      stream: true, // Required for streaming chunks
    }),
  });

  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

  const decoder = new TextDecoder();
  let buffer = '';

  // response.body is an AsyncIterable in modern JS (2025)
  for await (const chunk of response.body) {
    // 1. Decode binary chunk and add to buffer
    // { stream: true } handles multi-byte characters split between chunks
    buffer += decoder.decode(chunk, { stream: true });

    // 2. Process all complete JSON objects in the buffer
    const lines = buffer.split('\n');
    
    // Keep the last partial line in the buffer
    buffer = lines.pop();

    for (const line of lines) {
      if (line.trim() === '') continue;
      
      try {
        const json = JSON.parse(line);
        
        // 3. Handle 'thinking' (reasoning) or 'content' based on model
        if (json.message?.thinking) {
          process.stdout.write(`[Thinking]: ${json.message.thinking}`);
        } else if (json.message?.content) {
          process.stdout.write(json.message.content);
        }

        // Check if generation is complete
        if (json.done) {
          console.log('\n\n--- Generation Complete ---');
        }
      } catch (err) {
        console.error('Failed to parse line:', line, err);
      }
    }
  }
}

streamOllama().catch(console.error);

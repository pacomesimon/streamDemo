FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Ollama server
RUN sudo curl -fsSL https://ollama.com/install.sh | sh 

# Copy requirements and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY --chown=user . /app

# Start Ollama in the background, pull the model, then start the Gradio app
RUN nohup ollama serve > output1.log 2>&1 &
RUN ollama pull amsaravi/medgemma-4b-it:q8
CMD python app.py --port 7860

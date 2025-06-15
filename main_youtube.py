# main_youtube.py – Backend pipeline for YouTube video processing

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
import whisper
import tempfile
import subprocess
import uuid

# --- Initialization ---
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"  # Comment/remove if using native OpenAI keys
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    whisper_model = None
    print(f"[ERROR] Whisper model failed to load: {e}")

# --- Helper functions ---
def download_youtube_audio(url):
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--output", temp_path,
        url
    ]
    subprocess.run(command, check=True)
    return temp_path

def correct_transcript(raw_text):
    prompt = f"""You are a professional transcript editor.

Correct the following transcript by:
- Fixing grammar, punctuation, and structure
- Removing filler words (like 'um', 'uh', etc.)
- Making the transcript readable

Transcript:
{raw_text}
"""
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-qwen-32b:free",
        messages=[
            {"role": "system", "content": "You are a transcription corrector."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def summarize_text(text):
    prompt = f"""Summarize the following transcript into organized, sectioned notes.

Instructions:
- Use clear **section headings** (bold)
- List 3–5 bullet points per section
- Group similar ideas together
- Be concise and avoid repetition

Transcript:
{text}
"""
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-qwen-32b:free",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def polish_bullets(summary):
    prompt = f"""Polish the following grouped bullet points to improve structure, clarity, and formatting.

Guidelines:
- Keep section headers bolded
- Use clear, concise bullets
- Ensure spacing and formatting is consistent
-Make sure the bullet points are considerably smaller and easy to memorize than the summary. 
- Do not use has. Only use bullet points to mark the points.

Content:
{summary}
"""
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-qwen-32b:free",
        messages=[
            {"role": "system", "content": "You are a bullet point refiner."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- Request schema ---
class YouTubeRequest(BaseModel):
    url: str

# --- Main YouTube processing endpoint ---
@app.post("/api/process/youtube")
async def process_youtube(req: YouTubeRequest):
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    try:
        # Download audio
        audio_path = download_youtube_audio(req.url)

        # Transcribe
        result = whisper_model.transcribe(audio_path)
        raw_transcript = result["text"].strip()
        os.remove(audio_path)

        # Handle poor input
        if not raw_transcript or len(raw_transcript.split()) < 5:
            return {
                "polished_summary": "The audio is too corrupted or noisy to extract useful content."
            }

        # AI pipeline
        corrected = correct_transcript(raw_transcript)
        summary = summarize_text(corrected)
        polished = polish_bullets(summary)

        return {
            "summary": summary,
            "polished_summary": polished
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube pipeline failed: {str(e)}")

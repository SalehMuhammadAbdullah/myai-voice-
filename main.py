import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


class TTSRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/tts")
def generate_voice(request: TTSRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if len(text) > 4000:
        raise HTTPException(
            status_code=400,
            detail="Text is too long. Please use less than 4000 characters."
        )

    filename = f"{uuid.uuid4()}.mp3"
    output_file = AUDIO_DIR / filename

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_file)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"url": f"/audio/{filename}"}


app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

FROM python:3.10-slim

# 🔧 Add this line to install git and ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_youtube:app", "--host", "0.0.0.0", "--port", "8000"]

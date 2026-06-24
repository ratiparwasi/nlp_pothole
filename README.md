# NLP Pothole Detection

## Install

pip install -r requirements.txt

## Run
open anaconda prompt as administrator
switch directory with 'cd C:\coding\kep python\nlp_pothole'

uvicorn main:app --reload
or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## API

POST /chat

Request:

{
    "message": "Apa itu YOLO?"
}

Response:

{
    "response": "YOLO (You Only Look Once) adalah algoritma object detection yang digunakan untuk mendeteksi jalan berlubang dengan cepat."
}
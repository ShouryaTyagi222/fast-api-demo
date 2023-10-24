# pip install uvicorn gunicorn fastapi pydantic pandas

# to run `uvicorn api:app --reload`

# to open swagger `http://127.0.0.1:8000/docs`

# IMPORTING THE LIBRARIES
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import pytesseract
import requests

app=FastAPI()

class Item(BaseModel):
    image: list
    config : dict

@app.get('/get')
async def get_func():
    return {'API for OCR'}

@app.post('/')
async def scoring_endpoint(item:Item):

    # Input Extraction image_url and language
    item=item.dict()
    image_url=item['image'][0]['imageUri']
    language=item['config']['languages'][0]['sourceLanguage']

    # Image Loading
    img = Image.open(requests.get(image_url, stream=True).raw)
    
    # OCR on the image
    output= pytesseract.image_to_string(img,config='--oem 3 --psm 6',lang='eng')

    # Sending the Response
    return {'output':[{'source':output}]}
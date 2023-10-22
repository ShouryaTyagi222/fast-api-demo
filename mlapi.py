# pip install uvicorn gunicorn fastapi pydantic pandas

# to run `uvicorn mlapi:app --reload`

# to open swagger `http://127.0.0.1:8000/docs`

# IMPORTING THE LIBRARIES
from fastapi import FastAPI
from pydantic import BaseModel
import pytesseract as pt
from PIL import Image
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
    output= pt.image_to_string(img)
    message='success'
    status=200

    # Sending the Response
    return {'output':[{'source':output}],'status':{'statusCode':status,'message':message}}




# to deploy on deta-space
# iwr https://get.deta.dev/space-cli.ps1 -useb | iex

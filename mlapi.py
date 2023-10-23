# pip install uvicorn gunicorn fastapi pydantic pandas

# to run `uvicorn mlapi:app --reload`

# to open swagger for the api `http://127.0.0.1:8000/docs`

# IMPORTING THE LIBRARIES
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import requests

app=FastAPI()

# get doctr objects
def get_doctr_objs(json_op):
    result=''
    for page in json_op['pages']:
        for block in page['blocks']:
            for lines in block['lines']:
                for word in lines['words']:
                    result+=word['value']+' '
                result+='\n'
    return result

class Item(BaseModel):
    image: list
    config : dict

@app.get('/get')
async def get_func():
    return {'API for OCR'}

@app.post('/')
async def scoring_endpoint(item:Item):

    # Input Extraction: image_url and language
    item=item.dict()
    image_url=item['image'][0]['imageUri']
    language=item['config']['languages'][0]['sourceLanguage']
    temp_file_path='temp_image.png'   # creating a temp image on the image_url to be used with model

    # opening the image and saving
    img = Image.open(requests.get(image_url, stream=True).raw)
    img.save(temp_file_path)

    # OCR on the image
    doc = DocumentFile.from_images(temp_file_path)
    model = ocr_predictor(pretrained=True)
    result = model(doc)
    output = result.export()

    # preparing the output
    output=get_doctr_objs(output)
    print(output)

    # Sending the Response
    return {'output':[{'source':output}]}
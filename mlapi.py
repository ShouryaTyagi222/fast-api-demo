# pip install uvicorn gunicorn fastapi pydantic pandas

# to run `uvicorn mlapi:app --reload`

# to open swagger `http://127.0.0.1:8000/docs`

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
    imageUri: list
    config : dict

@app.get('/get')
async def get_func():
    return {'API for OCR'}

@app.post('/')
async def scoring_endpoint(item:Item):

    # Input Extraction image_url and language
    item=item.dict()
    image_url=item['imageUri'][0]
    language=item['config']['language']['sourceLanguage']
    file_path='temp_image.png'

    img = Image.open(requests.get(image_url, stream=True).raw)
    img.save(file_path)

    # OCR on the image
    doc = DocumentFile.from_images(file_path)

    # Create an OCR predictor object with the desired architectures
    model = ocr_predictor(pretrained=True)

    # Run the model on the document and get the output
    result = model(doc)
    output = result.export()

    # Print the output or visualize it
    output=get_doctr_objs(output)
    print(output)
    message='success'
    status=200

    # Sending the Response
    return {'output':[{'source':output}]}
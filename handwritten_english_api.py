# pip install uvicorn gunicorn fastapi pydantic pandas

# to run `uvicorn english_api:app --reload`

# to open swagger for the api `http://127.0.0.1:8000/docs`

# IMPORTING THE LIBRARIES
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PIL import Image
import requests

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.models.predictor import OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor

app=FastAPI()

def initialize_handwritten_models(language_model):
    det_model = db_resnet50(pretrained=True)
    det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)

    #Recognition model
    reco_model = crnn_vgg16_bn(pretrained=False, vocab='ॲऽऐथफएऎह८॥ॉम९ुँ१ं।षघठर॓ॼड़गछिॱटऩॄऑवल५ढ़य़अञसऔयण॑क़॒ौॽशऍ॰ूीऒॊख़उज़ॻॅ३ओऌळनॠ०ेढङ४़ॢग़पऊॐज२डैभझकआदबऋखॾ॔ोइ्धतफ़ईृःा६चऱऴ७-')
    reco_param = torch.load(r'E:\IITB_OCR\crnn_vgg16_bn_handwritten_hindi.pt', map_location="cpu")
    reco_model.load_state_dict(reco_param)
    reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=1, mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)), reco_model)

    predictor = OCRPredictor(det_predictor, reco_predictor)

    return predictor

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

def predict(file_path):
    doc = DocumentFile.from_images(file_path)
    model=initialize_handwritten_models()
    result = model(doc)
    output = get_doctr_objs(result.export())
    return output


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
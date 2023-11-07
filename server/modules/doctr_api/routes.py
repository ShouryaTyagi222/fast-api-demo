import uuid
from typing import List

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .dependencies import save_uploaded_images
from .helper import (process_image, process_image_craft,
                     process_image_worddetector, process_multiple_image_craft,
                     process_multiple_image_doctr,
                     process_multiple_image_doctr_v2,
                     process_multiple_image_worddetector, save_uploaded_image)
from .models import LayoutImageResponse, ModelChoice
from .post_helper import process_dilate, process_multiple_dilate

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)


@router.post('/', response_model=List[LayoutImageResponse])
async def doctr_layout_parser(
	folder_path: str = Depends(save_uploaded_images),
	model: ModelChoice = Form(ModelChoice.doctr),
	dilate: bool = Form(False),
):
    ret = process_multiple_image_doctr_v2(folder_path)
    return ret
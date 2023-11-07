import os
import shutil
import time
import uuid
from collections import OrderedDict
from os.path import join
from subprocess import check_output, run
from tempfile import TemporaryDirectory
from typing import List, Tuple

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import UploadFile

from ..core.config import IMAGE_FOLDER
from .model import *
from ..textron.main import textron_main

# TODO: remove this line and try to set the env from the docker-compose file.

def logtime(t: float, msg:  str) -> None:
	print(f'[{int(time.time() - t)}s]\t {msg}')

t = time.time()

logtime(t, 'Time taken to load the doctr model')


def save_uploaded_image(image: UploadFile) -> str:
	"""
	function to save the uploaded image to the disk

	@returns the absolute location of the saved image
	"""
	t = time.time()
	print('removing all the previous uploaded files from the image folder')
	os.system(f'rm -rf {IMAGE_FOLDER}/*')
	location = join(IMAGE_FOLDER, '{}.{}'.format(
		str(uuid.uuid4()),
		image.filename.strip().split('.')[-1]
	))
	with open(location, 'wb+') as f:
		shutil.copyfileobj(image.file, f)
	logtime(t, 'Time took to save one image')
	return location

def process_images_textron(folder_path: str) -> List[LayoutImageResponse]:
	t = time.time()
	logtime(t, 'Time taken to load the doctr model')
	pages=textron_main(folder_path)

	t = time.time()
	logtime(t, 'Time taken to perform doctr inference')

	t = time.time()
	ret = []	
	for page in pages:
		regions = []
		for bbox in page:
			regions.append(
					Region.from_bounding_box(
						BoundingBox(x=bbox[2],y=bbox[3],w=bbox[4],h=bbox[5]),
						label=bbox[0],
					)
				)
		ret.append(
			LayoutImageResponse(
				regions=regions.copy()
			)
		)
	logtime(t, 'Time taken to process the doctr output')
	return ret
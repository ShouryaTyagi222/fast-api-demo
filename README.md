## Requirements
```
pip install uvicorn gunicorn fastapi pydantic pandas
```
## To Run on Local HOST
```
uvicorn mlapi:app --reload  # mlapi: name of the py file of api
```
## To open Swagger for localhost
```
http://127.0.0.1:8000/docs
```

## NOTE:
`mlapi.py`: is doctr model api
`api.py`: is the demo api
`request_ocr.json`: is the sample request format for ulca ocr
`response_ocr.json`: is the sample response format for ulca ocr
`sample_json_ocr.json`: is the sample json format to be submitted to ulca for adding the model

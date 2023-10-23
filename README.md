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

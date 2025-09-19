import os
import sys
import pandas as pd
import numpy as np
import certifi
ca = certifi.where()
import pymongo
from dotenv import load_dotenv
load_dotenv()

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_pickle_obj
from networksecurity.constants import training_pipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run
from fastapi.responses import Response  
from starlette.responses import RedirectResponse

MONGODB_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(MONGODB_URI, tlsCAFile=ca)

database = client[training_pipeline.DATA_INGESTION_DATABASE_NAME]
collection = database[training_pipeline.DATA_INGESTION_COLLECTION_NAME]

application = FastAPI()
app = application

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline_obj = TrainingPipeline()
        train_pipeline_obj.run_pipeline()
        return Response("Training successful!!")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.get("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_pickle_obj(file_path=training_pipeline.FINAL_PREPROCESSOR_PATH)
        model = load_pickle_obj(file_path=training_pipeline.FINAL_MODEL_PATH)
        network_model = NetworkModel(model= model, preprocessor=preprocessor)
        y_pred = network_model.predict(x=df)
        df["predicted_output"] = y_pred

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    try:
        run(app,host="localhost", port=8000)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
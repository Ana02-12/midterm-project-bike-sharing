"""FastAPI service for the bike demand model.

Run from the project root:
    uvicorn predict:app --reload

Interactive documentation: http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'bike_demand_model.joblib'
CONFIG_PATH = ROOT / 'models' / 'model_config.json'
STATIC_DIR = ROOT / 'static'
# Filled once at startup (see lifespan below)
artifacts = {}


@asynccontextmanager
async def lifespan(app):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(
            f"The model was saved with scikit-learn {config['sklearn_version']}, "
            f"but {sklearn.__version__} is installed. Install the same version."
        )
    artifacts['model'] = joblib.load(MODEL_PATH)
    artifacts['features'] = config['features']
    yield
    artifacts.clear()


app = FastAPI(
    title='Bike Sharing Demand API',
    description=(
        'Predicts the number of bike rentals (cnt) in one hour. '
        'The model was trained on 2011-2012 data, so yr accepts only 0 (2011) '
        'and 1 (2012): it predicts the demand level of these two years and '
        'cannot predict growth beyond them.'
    ),
    lifespan=lifespan,
)

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / 'index.html')


class BikeRequest(BaseModel):
    season: int = Field(ge=1, le=4, description='1 Winter, 2 Spring, 3 Summer, 4 Fall')
    yr: int = Field(ge=0, le=1, description='0 = 2011, 1 = 2012')
    mnth: int = Field(ge=1, le=12, description='Month, 1-12')
    hr: int = Field(ge=0, le=23, description='Hour of the day, 0-23')
    holiday: int = Field(ge=0, le=1, description='1 if the day is a holiday')
    weekday: int = Field(ge=0, le=6, description='0 = Sunday, ..., 6 = Saturday')
    workingday: int = Field(ge=0, le=1, description='1 if Monday-Friday and not a holiday')
    weathersit: int = Field(
        ge=1, le=4,
        description='1 Clear, 2 Mist/Cloudy, 3 Light Rain/Snow, 4 Heavy Rain/Snow (4 is treated as 3)',
    )
    temp: float = Field(ge=0, le=1, description='Normalized temperature, 0-1')
    hum: float = Field(ge=0, le=1, description='Normalized humidity, 0-1')
    windspeed: float = Field(ge=0, le=1, description='Normalized wind speed, 0-1')

    @model_validator(mode='after')
    def check_workingday(self):
        # In the training data workingday = 1 exactly for Monday-Friday that are not holidays
        expected = int(1 <= self.weekday <= 5 and self.holiday == 0)
        if self.workingday != expected:
            raise ValueError(
                f'workingday={self.workingday} does not match weekday={self.weekday} '
                f'and holiday={self.holiday}: expected workingday={expected}'
            )
        return self

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'season': 3, 'yr': 1, 'mnth': 7, 'hr': 17, 'holiday': 0,
                'weekday': 5, 'workingday': 1, 'weathersit': 1,
                'temp': 0.7, 'hum': 0.5, 'windspeed': 0.2,
            }]
        }
    }


class PredictionResponse(BaseModel):
    predicted_cnt: float


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': 'model' in artifacts,
        'sklearn_version': sklearn.__version__,
    }


@app.post('/predict', response_model=PredictionResponse)
def predict(bike: BikeRequest):
    row = bike.model_dump()
    if row['weathersit'] == 4:  # category 4 was merged into 3 in the training data
        row['weathersit'] = 3
    X = pd.DataFrame([row])[artifacts['features']]
    prediction = float(artifacts['model'].predict(X)[0])
    return PredictionResponse(predicted_cnt=round(prediction, 1))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Mock data for config and FORBIDDEN_KEYS
config = {          "ARTIFACT_DIR": './artifact/',
          "MODEL_NAME": 'cadets3_models',
          "LOSS_FACTOR": 1.5,
          "MAX_AVG_LOSS": 10,
          "MIN_AVG_LOSS": 4.7}
FORBIDDEN_KEYS = ["SECRET_KEY"]

class Command(BaseModel):
    api_args: dict

@app.get("/ping")
def ping():
    return {"status": "200 OK", "msg": "pong!\n"}

@app.get("/api/{cmd}/{begin_time}/{end_time}")
def kids_api(cmd: str, begin_time: str, end_time: str):
    cmd = cmd.lower()
    if cmd not in ("run", "test", "analyse", "investigate"):
        raise HTTPException(status_code=400, detail=f"To {cmd} is not allowed")
    return {"status": "200 OK", "msg": f"{cmd} successfully"}

@app.get("/config/update/{key}/{value}")
def update(key: str, value: str):
    key = key.upper()
    if key not in config.keys() or key in FORBIDDEN_KEYS:
        raise HTTPException(status_code=400, detail=f"Key {key} is not allowed")
    config[key] = value
    return {"status": "200 OK", "msg": f"{key} has been updated to {value}"}

@app.get("/config/view")
def view():
    data = {k: v for k, v in config.items() if k not in FORBIDDEN_KEYS}
    return {"status": "200 OK", "config": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
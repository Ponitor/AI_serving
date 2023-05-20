import numpy as np 
import os 
import uvicorn


from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
import shutil
from torchvision.transforms import functional as F

from emotic import Emotic
from yolo_inference import yolo_img

emotion_app = FastAPI()

model = "./models"
result_path = "./results"

@emotion_app.post("/infer/{customerId}")
async def infer(customerId:int, file: UploadFile = File(...)):
  
    UPLOAD_DIR = './uploadImage'

    if file != None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 디렉토리 생성
        local_path = os.path.normpath(os.path.join(UPLOAD_DIR, file.filename))
        print("local_path")
        print(local_path)
        with open(local_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    results = yolo_img(local_path,result_path,model,0,customerId) #모델에 이미지 넣기

      
    return {  "results" : results }   #"results" 안넣으며 에러



if __name__ == '__main__':
    app_str = 'emotionmodel:emotion_app'
    uvicorn.run(app_str, host='0.0.0.0', port=8000, reload=True, workers=1)


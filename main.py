from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
import pathlib
import PIL
from io import BytesIO
from diases import data
from predict_func import image_predict
import uvicorn

app = FastAPI()

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.post("/predict")
def upload(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        local_image_path = f"uploads/{file.filename}"
        with open(local_image_path, 'wb') as f:
            f.write(file.file.read())

        result = image_predict(local_image_path)

        return {
            "bashorat": f"{result['kasallik']}",
            "ehtimolligi": f"{result['ehtimolligi']}%",
            "maslahat": f"{result['tashxis']}\n{result['qushimcha']}"
        }
    except Exception as e:
        return {"message": f"There was an error: {str(e)}"}
    finally:
        # No need to close the file, FastAPI handles it
        pass

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0', reload=True)

from fastai.vision.all import *
import pathlib
from diases import quchimcha, data, tashxis, urlss
import cv2
from ultralytics import YOLO

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model  = YOLO("models/best.pt")
model_plant = load_learner("models/plant_name_classes.pkl")
model_apple = load_learner("models/apple_diasses.pkl")
model_chery = load_learner("models/Chery_diasses.pkl")
model_grape = load_learner("models/grape_diasses1.pkl")
model_peach = load_learner("models/peach_diasses.pkl")
model_pepper = load_learner("models/pepper_diasses1.pkl")
model_potato = load_learner("models/potato_diasses1.pkl")
models_stawberry = load_learner("models/stawberry_diasses1.pkl")
model_tomato = load_learner("models/tomato_diasses.pkl")
model_corn = load_learner("models/corn_diasses.pkl")


def image_predict(image_path):
    d = {}
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = []
    results = model.predict(img, stream=True)

    largest_area = 0
    largest_box = None

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            area = (r[2] - r[0]) * (r[3] - r[1])  # Calculate area of the box
            if area > largest_area:
                largest_area = area
                largest_box = box

    if largest_box is not None:
        r = largest_box.xyxy[0].astype(int)
        print(r)
        cv2.rectangle(img, r[:2], r[2:], (255, 0, 0), 2)

    r = largest_box.xyxy[0].astype(int)

    mask = np.zeros_like(img[:, :, 0])

    mask[r[1]:r[3], r[0]:r[2]] = 255

    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("images/rasm_de.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    pred, prob_id, prob = model_plant.predict(img)
    if pred == "apple":
        pred, prob_id, prob = model_apple.predict(img)

    elif pred == "Cherry":
        pred, prob_id, prob = model_chery.predict(img)

    elif pred == "Corn":
        pred, prob_id, prob = model_corn.predict(img)

    elif pred == "Grape":
        pred, prob_id, prob = model_grape.predict(img)

    elif pred == "peach":
        pred, prob_id, prob = model_peach.predict(img)

    elif pred == "pepper":
        pred, prob_id, prob = model_pepper.predict(img)

    elif pred == "potato":
        pred, prob_id, prob = model_potato.predict(img)

    elif pred == "strawberry":
        pred, prob_id, prob = models_stawberry.predict(img)

    elif pred == "tomato":
        pred, prob_id, prob = model_tomato.predict(img)

    predicted_class = data.get(pred, "Unknown")

    d['kasallik'] = predicted_class
    d['ehtimolligi'] = round(float((prob[prob_id])*100),2)
    d['tashxis'] = tashxis[pred]
    d['qushimcha'] = quchimcha
    d['urll'] = urlss[pred]
    return d

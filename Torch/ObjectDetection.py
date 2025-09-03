import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import time

# To load it elsewhere:
model = models.resnet18()
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # Assuming 2 classes: Chihuahua and Muffin
model.load_state_dict(torch.load('resnet18_chihuahua_muffin.pth'))
model.eval()

cap = cv.VideoCapture(0)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Chihuahua', 'Muffin']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_batch)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds[0]]

    cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Webcam - Chihuahua vs Muffin', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
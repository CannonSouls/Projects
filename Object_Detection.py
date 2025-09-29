import cv2
import torch
from torchvision import transforms

# Load your PyTorch model
model = torch.load('muffin_dog_model', weights_only=False, map_location=torch.device('cuda'))
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess frame for model using torchvision transforms

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = preprocess(img).unsqueeze(0)
    img = img.to(torch.device('cuda'))

    # Run inference
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    # Display prediction on frame
    label = 'Muffin' if pred == 1 else 'Chihuahua'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
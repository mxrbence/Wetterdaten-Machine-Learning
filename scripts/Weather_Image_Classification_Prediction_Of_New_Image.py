import kagglehub
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


path = "./Weather_images_new"

# ------------- Define the classes -------------

classes = ["Cloudy", "Rainy", "Sunshine", "Sunrise"]

label_map = {
    "Cloudy": 0,
    "Rainy": 1,
    "Sunshine": 2,
    "Sunrise": 3}

# ------------- Create Data list -------------

samples = []

for file in os.listdir(path):

    samples.append(os.path.join(path, file))

print("Anzahl Bilder:", len(samples))


# -------------- Define Dataset class -------------

class WeatherDataset(Dataset):
    def __init__(self, samples, image_size=128):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        image_path = self.samples[idx]
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(image_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype("float32") / 255
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        return img, image_path

# -------------- Create Dataset instances -------------

new_dataset = WeatherDataset(samples)

# -------------- Create DataLoader instances -------------

new_loader = DataLoader(
    new_dataset,
    batch_size=1,
    shuffle=False)

images = next(iter(new_loader))

# -------------- Define the CNN model -------------

model = nn.Sequential(

    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),           # Eindimensionale Vektor erstellen

    nn.Linear(128 * 16 * 16, 256),
    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(256, 4))

# -------------- Prediction of new image -------------

model.load_state_dict(torch.load("./best_weather_image_classification_cnn.pth", map_location="cpu"))

model.eval()

images_to_plot = []

titles = []

with torch.no_grad():

    for images, image_paths in new_loader:

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predicted = outputs.argmax(dim=1)

        confidence = probabilities.max(dim=1).values

        prediction = classes[predicted.item()]

        conf = confidence.item() * 100

        # Bild für Matplotlib vorbereiten
        img = images[0].permute(1, 2, 0).numpy()

        images_to_plot.append(img)

        titles.append(f"{prediction}\n{conf:.1f}%")

# ---------------- Plot aller Bilder ----------------

num_images = len(images_to_plot)

cols = 4                           # Bilder pro Zeile
rows = (num_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

# Falls nur eine Zeile existiert
axes = np.array(axes).reshape(-1)

for i in range(num_images):

    axes[i].imshow(images_to_plot[i])

    axes[i].set_title(titles[i], fontsize=10)

    axes[i].axis("off")

# Leere Felder ausblenden
for i in range(num_images, len(axes)):
    axes[i].axis("off")

plt.tight_layout()

plt.show()

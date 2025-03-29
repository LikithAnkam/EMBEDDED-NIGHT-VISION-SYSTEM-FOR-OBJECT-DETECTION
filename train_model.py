import os
import torch
import torchvision
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

# ✅ Custom Dataset to Read Pascal VOC XML Annotations
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text  # Get class label
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Assuming all objects are "pedestrians" (Class ID = 1)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        annotation_path = os.path.join(self.annotation_dir, image_filename.replace(".jpg", ".xml"))

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.parse_xml(annotation_path)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target  # ✅ Correctly formatted target

# ✅ Define Transform
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# ✅ Load Dataset
dataset = VOCDataset(
    image_dir="C:/Users/likit/Downloads/SmallDataset/Train/Images",
    annotation_dir="C:/Users/likit/Downloads/SmallDataset/Train/Annotations",
    transform=transform
)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ✅ Train Faster R-CNN
def train_yolo():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, targets in train_loader:
            images = list(image for image in images)
            targets = [{ "boxes": t["boxes"], "labels": t["labels"] } for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)  # ✅ Pass correctly formatted targets
            loss = sum(loss for loss in loss_dict.values())  # ✅ Sum all loss values
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item()}")

    torch.save(model.state_dict(), "fasterrcnn_trained.pth")

# train_yolo()


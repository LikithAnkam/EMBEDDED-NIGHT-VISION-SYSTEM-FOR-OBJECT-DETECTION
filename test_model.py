import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

def test_yolo():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load("yolov2_trained.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    test_img_path = "temp.png"
    image = cv2.imread(test_img_path)
    image_resized = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_resized)

    print(output)

    for obj in output[0]['boxes']:
        x, y, w, h = obj.int().tolist()
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imwrite("output_test.jpg", image)
    cv2.imshow("Test Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

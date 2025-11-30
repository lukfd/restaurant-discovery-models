import argparse
import torch
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode
from torch import nn

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

CLASS_NAMES = ["low", "medium", "high"]  # 0-3 stars, 3.5-4 stars, 4.5-5 stars


def get_transforms(model_type: str):
    if model_type == "basic_cnn":
        return models.ResNet18_Weights.DEFAULT.transforms()
    elif model_type == "resnet18":
        return models.ResNet18_Weights.DEFAULT.transforms()
    elif model_type == "resnet50":
        return models.ResNet50_Weights.DEFAULT.transforms()
    elif model_type == "regnet_y_400mf":
        return models.RegNet_Y_400MF_Weights.DEFAULT.transforms()
    elif model_type == "regnet_y_8gf":
        return models.RegNet_Y_8GF_Weights.DEFAULT.transforms()
    else:
        return models.ResNet18_Weights.DEFAULT.transforms()


def get_model(model_type: str) -> nn.Module:
    if model_type == "basic_cnn":
        from basic_cnn import BasicCNN
        model = BasicCNN().to(DEVICE)
    elif model_type == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    elif model_type == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    elif model_type == "regnet_y_400mf":
        model = models.regnet_y_400mf()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    elif model_type == "regnet_y_8gf":
        model = models.regnet_y_8gf()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    else:
        raise Exception(f"{model_type} not defined")
    
    return model


def load_and_preprocess_image(image_path: str, transform):
    img = read_image(image_path, mode=ImageReadMode.RGB)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def make_prediction(model_path: str, image_path: str):
    # Extract model type from path
    model_type = model_path.split("/")[1]
    
    # Load model
    model = get_model(model_type)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    model.eval()
    
    # Get transforms for the model type
    transform = get_transforms(model_type)
    
    # Load and preprocess image
    img = load_and_preprocess_image(image_path, transform)
    img = img.to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        output = model(img)
        predicted_class = output.argmax(dim=1).item()
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Print results
    print(f"\nImage: {image_path}")
    print(f"Model: {model_path}")
    print(f"\nPredicted Class: {CLASS_NAMES[predicted_class]} (class {predicted_class})")
    print(f"\nClass Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {probabilities[i]:.4f}")
    
    return CLASS_NAMES[predicted_class]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make prediction on a single image using a trained model"
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Path to saved model", required=True
    )
    parser.add_argument(
        "-i", "--image", type=str, help="Path to image file", required=True
    )
    
    args = parser.parse_args()
    
    predicted_class = make_prediction(args.model, args.image)

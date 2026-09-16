"""
predict.py — Inference Pipeline for PawScan AI
================================================
Loads the trained EfficientNet-B0 model and runs inference on a single image.
Returns: predicted class, confidence score, all class probabilities.

USAGE:
    from predict import PawScanPredictor

    predictor = PawScanPredictor("pawscan_model.pth")
    result = predictor.predict("path/to/dog_photo.jpg")
    print(result)
"""

import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import numpy as np


class PawScanPredictor:
    """Load trained model and run inference on pet images."""

    def __init__(self, model_path, class_names_path=None, device=None,
                 disease_info_path=None):
        """
        Args:
            model_path: Path to pawscan_model.pth
            class_names_path: Path to class_names.json (optional, loaded from checkpoint if not provided)
            device: torch device (defaults to CUDA if available)
            disease_info_path: Path to data/disease_info.json (optional — used
                to look up per-class severity instead of the fallback map)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get class info from checkpoint
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path) as f:
                meta = json.load(f)
            self.class_names = meta["class_names"]
            self.idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
            self.display_names = meta.get("display_names", {})
        else:
            self.class_names = checkpoint.get("class_names", [])
            self.idx_to_class = checkpoint.get("idx_to_class", {})
            self.display_names = {}

        # Per-class severity from disease_info.json when available, with a
        # static fallback so inference never crashes if the file is missing.
        self._severity_map = {}
        if disease_info_path and os.path.exists(disease_info_path):
            try:
                with open(disease_info_path) as f:
                    info = json.load(f)
                self._severity_map = {
                    cls: d.get("severity", "moderate") for cls, d in info.items()
                }
            except (json.JSONDecodeError, OSError):
                pass

        self._default_severity_map = {
            "Healthy": "none",
            "Hypersensitivity": "mild",
            "Dermatitis": "moderate",
            "Fungal_infections": "moderate",
            "demodicosis": "severe",
            "ringworm": "severe"
        }

        num_classes = len(self.class_names)

        # Build model architecture (same as training)
        self.model = models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image transforms (same as validation — no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ PawScanPredictor loaded on {self.device}")
        print(f"   Classes: {self.class_names}")

    def predict(self, image_input):
        """
        Run inference on a single image.

        Args:
            image_input: Path to image file OR PIL Image object

        Returns:
            dict with:
                - predicted_class: raw class name (e.g. "ringworm")
                - display_name: friendly name (e.g. "Ringworm (Dermatophytosis)")
                - confidence: float (0-1)
                - confidence_pct: float (0-100)
                - all_probabilities: dict {class_name: probability}
                - is_healthy: bool
                - severity: str ("none", "mild", "moderate", "severe")
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get results
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = self.idx_to_class.get(predicted_idx.item(), str(predicted_idx.item()))
        display_name = self.display_names.get(predicted_class, predicted_class)

        # Build probability dict
        all_probs = {}
        for i, prob in enumerate(probabilities.cpu().numpy()):
            class_name = self.idx_to_class.get(i, str(i))
            all_probs[class_name] = round(float(prob), 4)

        # Determine severity (prefer disease_info.json, fall back to static map)
        severity = self._severity_map.get(
            predicted_class,
            self._default_severity_map.get(predicted_class, "moderate")
        )

        result = {
            "predicted_class": predicted_class,
            "display_name": display_name,
            "confidence": round(float(confidence), 4),
            "confidence_pct": round(float(confidence) * 100, 1),
            "all_probabilities": all_probs,
            "is_healthy": predicted_class == "Healthy",
            "severity": severity
        }

        return result

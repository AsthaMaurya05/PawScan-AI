"""
gradcam.py — Grad-CAM Explainability for PawScan AI
===================================================
Implements Grad-CAM (Selvaraju et al., 2017) for the EfficientNet-B0
classifier so users can see WHERE the model looked when making a
prediction — critical for trust in any health-related CV model.

The heatmap is computed on the final convolutional feature map
(7x7x1280 for a 224x224 input) of EfficientNet-B0:
    weights_k = global-average-pooled gradient of class score k
    CAM       = ReLU( sum_k weights_k * activation_k )
    heatmap   = CAM normalized to [0, 1], upscaled to image size

USAGE:
    from gradcam import generate_gradcam
    from predict import PawScanPredictor

    predictor = PawScanPredictor("models/pawscan_model.pth", "class_names.json")
    overlay, cam, class_name = generate_gradcam(predictor, pil_image)
    overlay.save("gradcam_overlay.png")

No extra dependencies — uses torch, PIL, numpy and matplotlib only.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


class PawScanGradCAM:
    """Grad-CAM for the PawScan EfficientNet-B0 model.

    Attach one instance to a loaded predictor (not per-scan — hooks are
    registered once in the constructor and reused).
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device

        # Final conv feature map of EfficientNet-B0:
        # model.features[-1] outputs (B, 1280, 7, 7) for 224x224 inputs.
        self.target_layer = self.model.features

        self._activations = None
        self._gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)

    # ── hook callbacks ─────────────────────────────────────────
    def _forward_hook(self, module, input, output):
        """Save the feature map; attach a gradient hook when grads are live."""
        self._activations = output.detach()
        # Registering a hook on the output tensor (only possible when it
        # requires grad) captures exactly the gradient flowing back into
        # this layer, without the noisy full-backward-hook warning.
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad.detach()

    # ── main API ────────────────────────────────────────────────
    def generate(self, image, class_idx=None):
        """Compute the Grad-CAM heatmap for one image.

        Args:
            image: PIL Image (RGB or convertible to it)
            class_idx: optional class index to explain; defaults to the
                model's own top prediction

        Returns:
            (cam, class_idx) where cam is a normalized HxW numpy array
            (same H, W as the input image) with values in [0, 1].
        """
        was_training = self.model.training
        self.model.eval()

        try:
            img_tensor = self.predictor.transform(image.convert("RGB"))
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Forward pass WITH gradients (predict() uses no_grad, so we
            # can't reuse it here).
            output = self.model(img_tensor)  # (1, num_classes)

            if class_idx is None:
                class_idx = int(torch.argmax(output, dim=1).item())

            score = output[0, class_idx]
            self.model.zero_grad(set_to_none=True)
            score.backward()

            activations = self._activations  # (1, C, H, W)
            gradients = self._gradients      # (1, C, H, W)

            # Channel importance = spatial mean of the gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps -> CAM
            cam = F.relu((weights * activations).sum(dim=1))  # (1, H, W)
            cam = cam.squeeze(0).cpu().numpy()

            # Normalize to [0, 1] (guard against all-zero CAM)
            if cam.max() > 0:
                cam = cam / cam.max()

            # Upscale to the original image resolution
            w, h = image.size
            cam_img = Image.fromarray((cam * 255).astype(np.uint8))
            cam_img = cam_img.resize((w, h), Image.BILINEAR)
            cam = np.asarray(cam_img, dtype=np.float32) / 255.0

            return cam, class_idx
        finally:
            self.model.train(was_training)

    def overlay(self, image, cam, alpha=0.4):
        """Blend the heatmap over the original photo.

        Args:
            image: PIL Image (RGB)
            cam: normalized HxW heatmap from generate()
            alpha: heatmap opacity (0=invisible, 1=solid)

        Returns:
            PIL Image (RGB) with the heatmap overlay.
        """
        base = image.convert("RGB")
        w, h = base.size

        if cam.shape[0] != h or cam.shape[1] != w:
            cam_img = Image.fromarray((cam * 255).astype(np.uint8))
            cam_img = cam_img.resize((w, h), Image.BILINEAR)
            cam = np.asarray(cam_img, dtype=np.float32) / 255.0

        # 'jet' colormap: blue (cool) -> red (hot / most important)
        colored = cm.jet(cam)[:, :, :3]  # HxWx3, floats in [0, 1]

        base_arr = np.asarray(base, dtype=np.float32)
        blended = (1 - alpha) * base_arr + alpha * colored * 255.0
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ─── module-level convenience ─────────────────────────────────
def generate_gradcam(predictor, image, class_idx=None, alpha=0.4):
    """One-shot helper (kept simple for demo/interview use).

    Returns:
        (overlay_image, cam, class_idx)
    """
    gradcam = PawScanGradCAM(predictor)
    cam, class_idx = gradcam.generate(image, class_idx=class_idx)
    overlay = gradcam.overlay(image, cam, alpha=alpha)
    return overlay, cam, class_idx

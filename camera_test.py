import pyrealsense2 as rs
# midfusion_train_complete_fixed.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.optim import AdamW
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
# ─── Mid‐Level Fusion Backbone ─────────────────────────────────────────────────

class MidLevelFusionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB ResNet50
        res_r = resnet50(pretrained=True)
        # Depth ResNet50 (single-channel)
        res_d = resnet50(pretrained=False)
        res_d.conv1 = nn.Conv2d(1, res_d.conv1.out_channels,
                                kernel_size=7, stride=2, padding=3, bias=False)

        return_layers = {"layer1":"c2", "layer2":"c3", "layer3":"c4", "layer4":"c5"}
        self.rgb_extractor   = IntermediateLayerGetter(res_r, return_layers)
        self.depth_extractor = IntermediateLayerGetter(res_d, return_layers)

        # FPN for concatenated features (C2–C5)
        in_chs = [256*2, 512*2, 1024*2, 2048*2]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_chs, out_channels=256)
        self.out_channels = 256

    def forward(self, x):
        # x may be a list of Tensors [4×H×W] or a single Tensor [B,4,H,W]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        # now x: [B,4,H,W]
        rgb   = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]

        feats_r = self.rgb_extractor(rgb)
        feats_d = self.depth_extractor(depth)
        fused   = {name: torch.cat([feats_r[name], feats_d[name]], dim=1)
                   for name in feats_r}
        return self.fpn(fused)
    
 # ─── Model Factory ─────────────────────────────────────────────────────────────

def build_midfusion_fasterrcnn(num_classes: int):
    backbone = MidLevelFusionBackbone()

    # 4 FPN levels → one size and aspect_ratios tuple per level
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5,1.0,2.0),) * 4
    )

    # ROI pooling on the 4 FPN maps: c2–c5
    roi_pool = MultiScaleRoIAlign(
        featmap_names=['c2','c3','c4','c5'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool
    )
    # Normalize 4th channel (depth) with zero-mean
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.0]
    model.transform.image_std  = [0.229, 0.224, 0.225, 1.0]
    return model

# Example class names (replace with your dataset classes)
COCO_CLASSES = {1: "buds", 2: "mouse", 3: "stepler"}  

def run_realsense_inference(model, device="cpu", score_threshold=0.5):
    # Configure Intel RealSense streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # rgb: [H,W,3], depth: [H,W]
            rgb_tensor = torch.tensor(rgb / 255., dtype=torch.float32).permute(2,0,1)  # [3,H,W]
            depth_tensor = torch.tensor(depth / 65535., dtype=torch.float32).unsqueeze(0)  # [1,H,W]

            # Concatenate along channel dimension → [4,H,W]
            input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(device)  # [1,4,H,W]

            # Forward pass
            model.eval()
            with torch.no_grad():
                predictions = model(input_tensor)[0]


            # Draw predictions
            rgb_copy = rgb.copy()
            if "boxes" in predictions and len(predictions['boxes']) > 0:
                for idx, box in enumerate(predictions['boxes']):
                    score = predictions['scores'][idx].item()
                    if score < score_threshold:
                        continue
                    class_id = predictions['labels'][idx].item()
                    class_name = COCO_CLASSES.get(class_id, "Unknown")
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cv2.rectangle(rgb_copy, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(rgb_copy, f"{class_name}: {score:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow("RGB + Detections", rgb_copy)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# Example usage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES=4
# Step 2: Recreate model architecture
print("Creating model architecture...")
model = build_midfusion_fasterrcnn(NUM_CLASSES)

# Step 3: Load trained weights
print("Loading trained weights...")
state_dict = torch.load("midfusion_best1.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✓ Model loaded successfully!")

run_realsense_inference(model, DEVICE)

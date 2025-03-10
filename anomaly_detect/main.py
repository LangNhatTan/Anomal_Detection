import os
import argparse
import logging
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from network.data_loader import SingleVideoIter
from network.feature_extractor import to_segments
from network.anomaly_detector_model import AnomalyDetector
from network.TorchUtils import get_torch_device
from utils.load_model import load_models
from utils.utils import build_transforms

# Environment settings for single-threaded processing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def anomaly_detect(
    source: str,
    feature_extractor_path: str = "./pretrained/c3d.pickle",
    ad_model_path: str = "./exps/c3d/models/epoch_80000.pt",
    feature_method: str = "c3d",
    n_segments: int = 30,
    output_dir: str = "output"
):
    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) else 24
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_video_path = os.path.join(output_dir, "output_video.avi")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    overall_start_time = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anomaly_detector, feature_extractor = load_models(
        feature_extractor_path,
        ad_model_path,
        features_method=feature_method,
        device=device,
    )
    n_segments = int(total_frames / fps)
    # Feature extraction
    data_loader = SingleVideoIter(
        clip_length=16,
        frame_stride=1,
        video_path=source,
        video_transform=build_transforms(mode=feature_method),
        return_label=False,
    )
    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logging.info("Extracting features...")
    features = torch.tensor([])

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_iter), total=len(data_iter)):
            outputs = feature_extractor(data.to(device)).detach().cpu()
            features = torch.cat([features, outputs])
    
    features = features.numpy()
    if features.ndim == 0 or features.size == 0:
        raise ValueError("No valid features were extracted. Please check the input video.")
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    features = to_segments(features, n_segments)

    # Anomaly detection
    logging.info("Performing anomaly detection...")
    features_tensor = torch.tensor(features).to(device)
    with torch.no_grad():
        predictions = anomaly_detector(features_tensor).detach().cpu().numpy().flatten()
    plt.plot(predictions)
    plt.show()
    



if __name__ == "__main__":
    source = "D:/tanlailaptrinhpython/Computer_Vision/venv_anomaly/video/sample5.mp4"
    anomaly_detect(source = source)

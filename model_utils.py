import numpy as np
import cv2
import torch
import os

def preprocess(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a NumPy array, but got {type(image)}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 110))
    image = image[13:97, :]

    return image.astype(np.float32) / 255.0


def stack_frames(stacked_frames, new_frame, is_new_episode):
    frame = preprocess(new_frame)

    if is_new_episode:
        stacked_frames = [frame] * 4
    else:
        stacked_frames.append(frame)
        stacked_frames.pop(0)

    return np.stack(stacked_frames, axis=0), stacked_frames


def load_model(filepath="model.pth"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at {filepath}")

    model = torch.load(filepath)
    model.eval()
    return model


def save_model(model, filepath="model.pth"):
    torch.save(model, filepath)

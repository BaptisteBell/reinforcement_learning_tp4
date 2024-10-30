import numpy as np
import cv2
import torch

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image to reduce the size and grayscales it.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 110))
    image = image[13:97, :]
    print(image.shape)
    return image / 255.0

def stack_frames(stacked_frames, new_frame, is_new_episode):
    frame = preprocess(new_frame)
    if is_new_episode:
        stacked_frames = [frame] * 4
    else:
        stacked_frames.append(frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames

def load_model():
    model = torch.load("model.pth")
    model.eval()
    return model

def save_model(model):
    torch.save(model, "model.pth")
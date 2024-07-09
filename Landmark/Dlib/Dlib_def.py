import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ランドマーク検出
def get_dlib_landmarks(img, cnn_face_detector, sp):
    try:
        # 顔検出
        dets = cnn_face_detector(img, 1)

        if len(dets) > 0:
            d = dets[0]
            # 68点のランドマーク
            shape = sp(img, d.rect)
            landmarks = np.array([[shape.part(j).x, shape.part(j).y] for j in range(shape.num_parts)])
            return d, landmarks
        else:
            return None, None
    except:
        return None, None
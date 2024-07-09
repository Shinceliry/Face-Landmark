from torch import e
import face_alignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ランドマーク検出
def get_fa_landmarks(img, fa):
    try:
        preds, scores, bboxes = fa.get_landmarks(img, return_bboxes=True)
        if len(preds) > 0:
            # bbox = bboxes[0].astype(np.float32)
            return bboxes[0], preds[0]
        else:
            return None, None
    except:
        return None, None
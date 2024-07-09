import numpy as np
import torch
import torch.nn.functional as F

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
import mmpz_model_setup as model

def get_heatmap(img, bbox, pose_results):
    # bboxの1.25倍の領域の画像を取得
    left, top, right, bottom = bbox[0]
    width = right - left
    height = bottom - top
    width2 = width * 1.25
    height2 = height * 1.25

    center = np.array([(left + right) / 2, (top + bottom) / 2])

    left2, top2 = center - np.array((width2, height2)) / 2
    right2, bottom2 = center + np.array((width2, height2)) / 2

    # ヒートマップの取得
    heatmaps = pose_results[0].pred_fields.heatmaps.cpu()
    heatmap, _ = heatmaps.max(dim=0)

    # 1. 画像全体の座標のmap i:0~height-1, j: 0~width-1
    map_i = np.arange(img.shape[0])
    map_j = np.arange(img.shape[1])

    # 2. (left, top) = (-1, -1), (right, bottom) = (1, 1)となるmapを作成
    map_y = (map_i - top2) * 2 / (height2) - 1
    map_x = (map_j - left2) * 2 / (width2) - 1
    yy, xx = torch.meshgrid(torch.Tensor(map_y), torch.Tensor(map_x))

    # 3. grid-sample
    grid_sample_input = heatmap.unsqueeze(0).unsqueeze(0)  # H, W --> N, C, H, W
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) # N, H, W, 2
    heatmap_img = F.grid_sample(grid_sample_input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return heatmap_img.squeeze(0).squeeze(0).numpy()

def get_mmpose_landmarks(img, inference_detector, pose_estimator):
    # predict bbox
    det_result = inference_detector(model.detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[nms(bboxes, model.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    
    #landmarks
    landmarks = pose_results[0].pred_instances.keypoints[0]
    
    #heatmap
    heatmap = get_heatmap(img, bboxes, pose_results)
    
    return bboxes, landmarks, heatmap
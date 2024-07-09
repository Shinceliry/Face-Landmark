import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

import mmpz_model_setup as model
import mmpz_def

import sys
sys.path.append('../')
from DrawLandmark import drawlandmark

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def parse_args():
    parser = argparse.ArgumentParser(description="Landmark detection and heatmap generation using mmpose.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output images.')
    parser.add_argument('--resolution', type=int, nargs=2, default=None, help='Desired resolution (width height).')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 画像の用意
    origin_img = cv2.imread(args.image)
    if args.resolution:
        desired_resolution = tuple(args.resolution)
        img = cv2.resize(origin_img, desired_resolution, interpolation=cv2.INTER_LINEAR)
    else:
        img = origin_img

    # ランドマーク検出
    bbox, mmpose_landmarks, mmpose_heatmap = mmpz_def.get_mmpose_landmarks(img, inference_detector, model.pose_estimator)

    # ランドマーク描画
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCVの画像はBGRフォーマットなので、RGBに変換
    drawlandmark.draw_landmarks(ax, mmpose_landmarks)

    '''
    # bboxの表示
    bbox_xy = bbox[0][:2]
    bbox_wh = bbox[0][2:] - bbox[0][:2]
    ax.add_patch(patches.Rectangle(xy=bbox[0][:2], width=bbox_wh[0], height=bbox_wh[1], fill=False, color='r'))
    '''

    # 画像を保存
    save_path1 = os.path.join(args.output, 'landmarks.jpg')
    plt.savefig(save_path1)
    print(f'Image saved at {save_path1}')

    # ヒートマップ描画
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCVの画像はBGRフォーマットなので、RGBに変換
    ax2.imshow(mmpose_heatmap, cmap='turbo', alpha=0.5)

    # 画像を保存
    save_path2 = os.path.join(args.output, 'heatmap.jpg')
    plt.savefig(save_path2)
    print(f'Image saved at {save_path2}')

if __name__ == '__main__':
    main()
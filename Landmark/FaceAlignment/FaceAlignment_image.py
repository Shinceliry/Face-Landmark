import face_alignment
import matplotlib.pyplot as plt
import cv2
import numpy as np
import FaceAlignment_def
import os
import argparse
import sys

# パスの設定
sys.path.append('../')
from DrawLandmark import drawlandmark

def parse_args():
    parser = argparse.ArgumentParser(description="Face landmark detection using FaceAlignment.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use for computation.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image.')
    parser.add_argument('--resolution', type=int, nargs=2, default=None, help='Desired resolution (width height).')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 検出器の用意
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device)

    # 画像の用意
    origin_img = cv2.imread(args.image)
    if args.resolution:
        desired_resolution = tuple(args.resolution)
        img = cv2.resize(origin_img, desired_resolution, interpolation=cv2.INTER_LINEAR)
    else:
        img = origin_img

    # ランドマークの検出
    bboxes, landmarks = FaceAlignment_def.get_fa_landmarks(img, fa)

    # 結果の表示
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCVの画像はBGRフォーマットなので、RGBに変換
    drawlandmark.draw_landmarks(ax, landmarks)

    # 画像を保存
    plt.savefig(args.output)
    print(f'Image saved at {args.output}')

if __name__ == '__main__':
    main()
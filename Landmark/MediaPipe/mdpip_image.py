import matplotlib.pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
import mdpip_def
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Face landmark detection using MediaPipe.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image.')
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

    # ランドマークの検出
    landmarks = mdpip_def.get_mp_landmarks(img)

    # 結果の表示
    fig, ax = plt.subplots()
    ax.axis('off')
    mdpip_def.draw_mp_landmarks(ax, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks)  # OpenCVの画像はBGRフォーマットなので、RGBに変換

    # 画像を保存
    plt.savefig(args.output)
    print(f'Image saved at {args.output}')

if __name__ == '__main__':
    main()
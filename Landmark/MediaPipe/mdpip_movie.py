import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib.style as mplstyle
import numpy as np
import mediapy
import mediapipe as mp
import mdpip_def
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Face landmark detection in video using MediaPipe.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate of the output video.')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 処理高速化オプション
    mplstyle.use('fast') 

    # 動画読み込み
    video = mediapy.read_video(args.video)

    # 結果の動画用のフレームリストを作成
    mediapipe_result_video = []

    # ランドマーク
    for frame_img in tqdm(video):
        # ランドマーク検出
        mediapipe_landmarks = mdpip_def.get_mp_landmarks(frame_img)
        
        # ランドマーク描画
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis('off')
        ax.imshow(frame_img)
        mdpip_def.draw_mp_landmarks(ax, frame_img, mediapipe_landmarks)
        
        # 描画した画像をフレームリストに追加
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        mediapipe_result_video.append(np.asarray(draw_image))
        fig.clear()
        plt.close(fig)
    
    # 生成したフレームリストから動画を作成
    mediapy.write_video(args.output, mediapipe_result_video, fps=args.fps)
    print(f'Video saved at {args.output}')

if __name__ == '__main__':
    main()
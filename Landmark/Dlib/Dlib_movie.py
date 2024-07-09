import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib.style as mplstyle
import numpy as np
import mediapy
import dlib
import Dlib_def
import os
import argparse
import sys

# パスの設定
sys.path.append('../')
from DrawLandmark import drawlandmark

def parse_args():
    parser = argparse.ArgumentParser(description="Face landmark detection in video using dlib.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--detector', type=str, required=True, help='Path to the mmod_human_face_detector.dat file.')
    parser.add_argument('--shape_predictor', type=str, required=True, help='Path to the shape_predictor_68_face_landmarks.dat file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate of the output video.')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 検出器の用意
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.detector)
    sp = dlib.shape_predictor(args.shape_predictor)

    # 処理高速化オプション
    mplstyle.use('fast') 

    # 動画読み込み
    video = mediapy.read_video(args.video)

    # 結果の動画用のフレームリストを作成
    dlib_result_video = []

    # ランドマーク
    for frame_img in tqdm(video):
        # ランドマーク検出
        _, dlib_landmarks = Dlib_def.get_dlib_landmarks(frame_img, cnn_face_detector, sp)
        
        # ランドマーク描画
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis('off')
        ax.imshow(frame_img)
        drawlandmark.draw_landmarks(ax, dlib_landmarks)
        
        # 描画した画像をフレームリストに追加
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        draw_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        dlib_result_video.append(np.asarray(draw_image))
        fig.clear()
        plt.close(fig)
    
    # 生成したフレームリストから動画を作成
    mediapy.write_video(args.output, dlib_result_video, fps=args.fps)
    print(f'Video saved at {args.output}')

if __name__ == '__main__':
    main()
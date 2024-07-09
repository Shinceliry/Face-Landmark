import mediapy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import mmpz_model_setup as model
import mmpz_def
import os
import argparse

import sys
sys.path.append('../')
from DrawLandmark import drawlandmark

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def parse_args():
    parser = argparse.ArgumentParser(description="Landmark detection and heatmap generation in video using mmpose.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output videos.')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate of the output videos.')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 処理高速化オプション
    mplstyle.use('fast') 

    # 動画の読み込み
    video = mediapy.read_video(args.video)

    # 個別の動画用のフレームリストを作成
    result_videos = {
        "mmpose": [],
        "mmpose_heatmap": []
    }

    # ランドマーク
    for frame_img in tqdm(video):
        _, mmpose_landmarks, mmpose_heatmap = mmpz_def.get_mmpose_landmarks(frame_img, inference_detector, model.pose_estimator)
        results = dict(
            mmpose = mmpose_landmarks,
            mmpose_heatmap = mmpose_heatmap
        )
        
        # 各ランドマーク検出結果に対して個別の画像を生成
        for label, result in results.items():
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.axis('off')
            ax.imshow(frame_img)
            if result is not None:
                if label == 'mmpose_heatmap':
                    ax.imshow(result, alpha=0.5, cmap='turbo')
                else:
                    drawlandmark.draw_landmarks(ax, result)
            fig.tight_layout()
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            draw_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            result_videos[label].append(np.asarray(draw_image))
            fig.clear()
            plt.close(fig)

    # 個別の動画として保存
    for label, frames in result_videos.items():
        output_path = os.path.join(args.output, f'{label}_movie.mp4')
        mediapy.write_video(output_path, frames, fps=args.fps)
        print(f'Video saved at {output_path}')

if __name__ == '__main__':
    main()
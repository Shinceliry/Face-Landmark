import locale
locale.getpreferredencoding = lambda: "UTF-8"

from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

det_config = './mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth'
device = 'cuda:0'
nms_thr = 0.3

# AFLW
# pose_config = "./mmpose/configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py"
# pose_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"

# 300w
pose_config = "./mmpose/configs/face_2d_keypoint/topdown_heatmap/300w/td-hm_hrnetv2-w18_8xb64-60e_300w-256x256.py"
pose_checkpoint = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256-eea53406_20211019.pth"
draw_heatmap = True

# build detector
detector = init_detector(
    det_config, det_checkpoint, device=device)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))
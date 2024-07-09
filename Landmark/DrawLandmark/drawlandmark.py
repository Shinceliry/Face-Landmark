import numpy as np

# Draw Landmark
# Dlib, FaceAlignment, MMposedeでは68個のランドマーク
def draw_landmarks(ax, landmarks):
    if landmarks is not None:
        parts = dict(
            jaw=slice(0, 17),
            right_eyebrow=slice(17, 22),
            left_eyebrow=slice(22, 27),
            nose_bridge=slice(27, 31),
            nose_tip=slice(31, 36),
            right_eye=slice(36, 42),
            left_eye=slice(42, 48),
            outer_lip=slice(48, 60),
            inner_lip=slice(60, 68)
        )

        for k, v in parts.items():
            if k in ['right_eye', 'left_eye', 'outer_lip', 'inner_lip']:
                x, y = landmarks[v].T
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            else:
                x, y = landmarks[v].T
            ax.plot(x, y, c='white')

        x, y = landmarks.T
        ax.scatter(x, y, s=10, marker="o", edgecolor='black', c='white')
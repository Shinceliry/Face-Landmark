#Dlib, Face-Alignment, MMposeのランドマークインデックス

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
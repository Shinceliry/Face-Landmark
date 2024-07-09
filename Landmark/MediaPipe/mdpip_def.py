import mediapipe as mp
import math

def get_mp_landmarks(img):
    face_mesh = mp.solutions.face_mesh
    face_detector = face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 検知処理
    results = face_detector.process(img)
    return results


def draw_mp_landmarks(ax, img, results):
    face_mesh = mp.solutions.face_mesh

    # mp.solutions.drawing_utils.draw_landmarks
    dst = img.copy()

    # 可視化
    drawing = mp.solutions.drawing_utils
    drawing_spec = drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0), )
    drawing_styles = mp.solutions.drawing_styles

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # drawing.draw_landmarks(
            #     image=dst,
            #     landmark_list=face_landmarks,
            #     connections=face_mesh.FACEMESH_TESSELATION,
            #     connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
            #     )
            drawing.draw_landmarks(
                image=dst,
                landmark_list=face_landmarks,
                # landmark_drawing_spec=None,
                connections=face_mesh.FACEMESH_CONTOURS,
                connection_drawing_spec=drawing.DrawingSpec(thickness=3, circle_radius=20, color=(255, 255, 255))
                # connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                )
    ax.imshow(dst)
    
# Convert the normalized position of a landmark into pixel coordinates
def get_landmark_position(landmark, image_width, image_height):
    landmark_x = int(landmark.x * image_width)
    landmark_y = int(landmark.y * image_height)
    
    return landmark_x, landmark_y

# Calculate the Euclidean distance between two points.
def calculate_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    return distance
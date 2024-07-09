import numpy as np

def get_landmark_position(landmarks, landmark_number):
    position = landmarks[landmark_number]
    print(landmark_number, "bannnoranndoma-kunozahyouha", position, "desu")
    
    return position

def get_landmark_distance(landmark1, landmark2):
    distance = np.linalg.norm(np.array(landmark1) - np.array(landmark2))
    print("LMDha", distance, "desu")
    
    return distance
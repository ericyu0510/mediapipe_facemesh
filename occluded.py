from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

_BLACK = (0, 0, 0)
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

# For static images:
image_path = 'C:\\Users\\ericy\\Desktop\\data512x512\\test\\'
IMAGE_FILES = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))]
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
occlude_color = _RED

# the landmark index below corresponds to the connection map provided by mediapipe
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
left_eye_landmarks = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
left_eyebrow_landmarks = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
right_eyebrow_landmarks = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
covered_landmarks = left_eye_landmarks + right_eye_landmarks + left_eyebrow_landmarks + right_eyebrow_landmarks

def resize_box(box, scale):
  '''
  The minimum area box is not preferable in our usage. Resize the original box to a proper size to occlude eyes
  and eyebrows.

  Args:
    box: the original minimum area box attained by cv2.minAreaRect()
    scale: to indicate the prefered size
  '''
  if(scale == 1):
    return np.int0(box)
  else:
    num_point = len(box)
    center = np.sum(box, axis=0)/num_point
    new_box = box + (box - center)*(scale-1)
    new_box = np.int0(new_box)
    return new_box

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_rows, image_cols, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      landmark_px_list = []
      for i in covered_landmarks:
        # convert normalized covered landmark to pixel landmark
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,
                                                   image_cols, image_rows)
        landmark_px_list.append(landmark_px)
      arr = np.array(landmark_px_list) # convert list to numpy array
      occluded_rect = cv2.minAreaRect(arr)
      occluded_box = cv2.boxPoints(occluded_rect)
      resized_occluded_box = resize_box(occluded_box, 1.5)
      cv2.fillPoly(annotated_image, [resized_occluded_box], occlude_color)

    cv2.imwrite('C:\\Users\\ericy\\Desktop/tmp/annotated_image' + str(idx) + '.png', annotated_image)

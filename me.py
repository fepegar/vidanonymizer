from pathlib import Path

import cv2
import click
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# Adapted from face_mesh.py
def get_oval_points(landmark_px):
    oval_points = {}
    OVAL = (
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    )
    for a, b in OVAL:
        if a in landmark_px:
            oval_points[a] = landmark_px[a]
        if b in landmark_px:
            oval_points[b] = landmark_px[b]
    return oval_points

def get_landmarks_pixels(image, landmark_list):
    # From drawing_utils.py
    VISIBILITY_THRESHOLD = 0.5
    PRESENCE_THRESHOLD = 0.5
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
                    landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                    landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            landmark.x, landmark.y,
            image_cols, image_rows,
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates

def draw_landmarks(image_bgr, landmarks_px, color=[30, 50, 190], radius=2, thickness=2):
    for landmark_px in landmarks_px:
        cv2.circle(image_bgr, landmark_px, radius, color, thickness)

def dilate_points(points, scaling):
    points = np.array(list(points), dtype=np.float32)
    center = points.mean(axis=0)
    points -= center
    points *= scaling
    points += center
    return points.astype(np.int32)

def get_face_mask(image_bgr, points, color=(255, 255, 255)):
    points = points.reshape(-1, 1, 2)  # https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html
    points = points.astype(np.int32)  # https://stackoverflow.com/a/18817152/3956024
    result = np.zeros(image_bgr.shape[:2])
    cv2.fillPoly(result, [points], color)
    return result

def get_bounding_box(mask):
    if mask.sum() == 0:
        return None
    where = np.where(mask)
    top, left = np.min(where, axis=1)
    bottom, right = np.max(where, axis=1) + 1  # avoid black lines
    return top, left, bottom, right

def get_subimage(image, bounding_box):
    top, left, bottom, right = bounding_box
    return image[top:bottom, left:right]

def replace_subimage(image, subimage, bounding_box, face_mask):
    top, left, bottom, right = bounding_box
    aux = np.zeros_like(image)
    subimage = subimage
    aux[top:bottom, left:right] = subimage
    face_mask = face_mask.astype(bool)
    image[face_mask] = aux[face_mask]

def blur_with_mask(image, face_mask, sigma=None):
    if sigma is None:
        sigma = image.shape[0] // 8 + 1
    bounding_box = get_bounding_box(face_mask)
    if bounding_box is None:
        return
    face = get_subimage(image, bounding_box)
    # blurred = cv2.GaussianBlur(face, (sigma, sigma), 0)
    blurred = cv2.blur(face, (sigma, sigma), 0)
    replace_subimage(image, blurred, bounding_box, face_mask)

def blur_face(image_bgr, landmarks_list):
    idx_to_coordinates = get_landmarks_pixels(image_bgr, landmarks_list)
    idx_to_coordinates_oval = get_oval_points(idx_to_coordinates)
    dilated_points = dilate_points(idx_to_coordinates_oval.values(), 2)
    face_mask = get_face_mask(image_bgr, dilated_points)
    blur_with_mask(image_bgr, face_mask)

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='input video file')
@click.option('--output', '-o', type=click.Path(), help='output video file')
@click.option('--plot/--no-plot', '-p', default=False, help='draw landmarks')
@click.option('--detconf', '-d', default=0.5, help='detection confidence')
@click.option('--traconf', '-t', default=0.5, help='tracking confidence')
@click.option('--maxfaces', '-m', default=2, help='maximum number of faces')
def main(input, output, plot, detconf, traconf, maxfaces):
    webcam = input is None
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0 if webcam else str(Path(input).expanduser()))

    # Check if stream opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    print(detconf)
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=detconf,
            min_tracking_confidence=traconf,
            max_num_faces=maxfaces) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                if webcam:
                    continue
                else:
                    break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            if webcam:
                image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    blur_face(image, face_landmarks)
                    if not plot:
                        continue
                    mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
            cv2.imshow('MediaPipe FaceMesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

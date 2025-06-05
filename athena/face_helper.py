import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

def crop_face_by_average_landmarks(frame_array, pose_landmarks, output_size=256, box_scale=5.0):
    """
    Crop and resize face region around the average of facial pose landmarks.
    Assumes frame_array is BGR (OpenCV format). Returns the cropped BGR face.

    Args:
        frame_array (np.ndarray): Original image in BGR (H×W×3).
        pose_landmarks (list of NormalizedLandmark): e.g. pose_results.pose_landmarks[0].
        output_size (int): Desired square output size (256).
        box_scale (float): Multiplier to enlarge bounding box around the face center.

    Returns:
        face_crop_resized_bgr (np.ndarray): Cropped-and-resized face in BGR.
        transform_info (dict): { 'x_min':…, 'y_min':…, 'scale':… } to map from crop→original.
    """
    h, w = frame_array.shape[:2]

    # Indices of the 11 face‐related pose landmarks: nose, eyes, ears, mouth corners.
    face_indices = list(range(11))

    # Collect pixel coordinates (in BGR frame) of those pose landmarks:
    face_points = np.array([
        [pose_landmarks[idx].x * w, pose_landmarks[idx].y * h]
        for idx in face_indices
    ])

    # Compute centroid (cx, cy)
    cx, cy = np.mean(face_points, axis=0)

    # Estimate average radius, then apply the box_scale factor:
    distances = np.linalg.norm(face_points - np.array([cx, cy]), axis=1)
    avg_radius = np.mean(distances)
    half_box = int(avg_radius * box_scale)

    x_min = int(max(0, cx - half_box))
    y_min = int(max(0, cy - half_box))
    x_max = int(min(w, cx + half_box))
    y_max = int(min(h, cy + half_box))

    # Crop from the original BGR frame:
    face_crop_bgr = frame_array[y_min:y_max, x_min:x_max]

    # Resize to (output_size × output_size) in BGR:
    face_crop_resized_bgr = cv2.resize(face_crop_bgr, (output_size, output_size))

    # Compute scale (assuming square):
    scale = (x_max - x_min) / output_size

    transform_info = {
        'x_min': x_min,
        'y_min': y_min,
        'scale': scale
    }
    return face_crop_resized_bgr, transform_info


def map_point_to_original_from_crop(x_crop, y_crop, transform_info):
    """
    Map a point in [0..output_size) crop space back to original-frame pixels.
    x_crop, y_crop are in crop‐pixel coordinates (not normalized).
    """
    x_orig = int(x_crop * transform_info['scale'] + transform_info['x_min'])
    y_orig = int(y_crop * transform_info['scale'] + transform_info['y_min'])
    return x_orig, y_orig


def draw_face_landmarks_on_image(rgb_image, detection_result):
    """
    Draw MediaPipe Face Mesh landmarks onto an RGB np.ndarray.
    Returns a new RGB np.ndarray with drawn landmarks.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        # Convert from list of NormalizedLandmark → NormalizedLandmarkList proto
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style()
        )

    return annotated


def map_landmarks_to_original(face_landmarks, transform_info, image_shape):
    """
    Given mediaPipe face_landmarks (list of NormalizedLandmark),
    map each landmark's (x, y) back to original-frame pixel coords.

    Args:
        face_landmarks: e.g. face_landmarker_result.face_landmarks (List[List[NormalizedLandmark]]).
        transform_info: from crop_face_by_average_landmarks.
        image_shape: original frame_array.shape (H, W, C).

    Returns:
        List of (x_orig, y_orig) tuples for each landmark in each detected face.
        For example, [[(x0,y0), (x1,y1), …],  … for each face detected].
    """
    mapped_all = []
    for face_pts in face_landmarks:
        mapped = []
        for lm in face_pts:
            # lm.x and lm.y are normalized in [0,1] over the 256×256 crop
            x_crop = lm.x * 256
            y_crop = lm.y * 256
            x_o, y_o = map_point_to_original_from_crop(x_crop, y_crop, transform_info)
            mapped.append((x_o, y_o))
        mapped_all.append(mapped)

    return mapped_all


def draw_mapped_face_landmarks_on_image(rgb_image, mapped_landmarks):
    """
    Draw mapped face landmarks onto an RGB image using MediaPipe drawing utilities.
    The landmarks should already be mapped to the original image space.

    Args:
        rgb_image (np.ndarray): The original RGB image to draw on.
        mapped_landmarks (list): List of (x,y) tuples in original image coordinates.

    Returns:
        np.ndarray: The image with face landmarks drawn.
    """
    if not mapped_landmarks:
        return rgb_image

    annotated = np.copy(rgb_image)
    h, w = annotated.shape[:2]

    # Convert mapped pixel coordinates to normalized landmarks
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(
            x=x/w,
            y=y/h,
            z=0  # We don't have z information in the mapped coordinates
        ) for x, y in mapped_landmarks[0]  # Use first face if multiple detected
    ])
    
    # Draw the face mesh components
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        face_landmarks_proto,
        mp.solutions.face_mesh.FACEMESH_TESSELATION,
        None,
        mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        face_landmarks_proto,
        mp.solutions.face_mesh.FACEMESH_CONTOURS,
        None,
        mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
    )
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        face_landmarks_proto,
        mp.solutions.face_mesh.FACEMESH_IRISES,
        None,
        mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
    )

    return annotated

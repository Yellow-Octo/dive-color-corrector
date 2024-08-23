import numpy as np
import cv2
import math

from core import applyFilter, getFilterValues

SAMPLE_SECONDS = 2  # Extracts color correction from every N seconds


def analyze_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0
    print("Analyzing...")
    while (cap.isOpened()):
        count += 1
        print(f"{count} frames", end="\r")
        ret, frame = cap.read()
        if not ret:
            if count >= frame_count:
                break
            if count >= 1e6:
                break
            continue
        if count % (fps * SAMPLE_SECONDS) == 0:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indexes.append(count)
            filter_matrices.append(getFilterValues(mat))
        yield count
    cap.release()
    filter_matrices = np.array(filter_matrices)
    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }


def process_video(video_data, yield_preview=False):
    cap = cv2.VideoCapture(video_data["input_video_path"])
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(video_data["output_video_path"], fourcc, video_data["fps"],
                                (int(frame_width), int(frame_height)))
    filter_matrices = video_data["filters"]
    filter_indices = video_data["filter_indices"]
    filter_matrix_size = len(filter_matrices[0])

    def get_interpolated_filter_matrix(frame_number):
        return [np.interp(frame_number, filter_indices, filter_matrices[..., x]) for x in range(filter_matrix_size)]

    print("Processing...")
    frame_count = video_data["frame_count"]
    count = 0
    cap = cv2.VideoCapture(video_data["input_video_path"])
    while (cap.isOpened()):
        count += 1
        percent = 100 * count / frame_count
        print("{:.2f}".format(percent), end=" % \r")
        ret, frame = cap.read()
        if not ret:
            if count >= frame_count:
                break
            if count >= 1e6:
                break
            continue
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        interpolated_filter_matrix = get_interpolated_filter_matrix(count)
        corrected_mat = applyFilter(rgb_mat, interpolated_filter_matrix)
        corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
        new_video.write(corrected_mat)
        if yield_preview:
            preview = frame.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[::, width:] = corrected_mat[::, width:]
            preview = cv2.resize(preview, (width, height))
            yield percent, cv2.imencode('.png', preview)[1].tobytes()
        else:
            yield None
    cap.release()
    new_video.release()

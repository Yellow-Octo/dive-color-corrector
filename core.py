import numpy as np
import cv2
import math

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2  # Extracts color correction from every N seconds


def correct_image(input_path, output_path):
    raw_data = cv2.imread(input_path)  # reads image into BGR format
    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)  # converts BGR to RGB
    corrected = generatedCorrected(rgb_data)
    cv2.imwrite(output_path, corrected)
    preview = raw_data.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected[::, width:]
    preview = cv2.resize(preview, (960, 540))
    return cv2.imencode('.png', preview)[1].tobytes()


def generatedCorrected(mat):
    original_mat = mat.copy()
    filterValues = getFilterValues(mat)
    corrected = applyFilter(original_mat, filterValues)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def redShift(mat, shiftAngle):
    U = math.cos(shiftAngle * math.pi / 180)
    W = math.sin(shiftAngle * math.pi / 180)

    # This uses a series of coefficients to adjust the RGB values based on a hue rotation matrix. These coefficients
    # are derived from the properties of the human vision system and standard color theory

    # 0.299: This is the luminance contribution from the red channel in the RGB to YIQ color space conversion
    # (YIQ is used in color television broadcasting). It signifies how much red contributes to the perceived brightness
    # of a color.
    # 0.701: This adjusts the red component’s contribution when shifting the hue, enhancing red as the angle increases.
    # 0.168: A smaller adjustment factor that slightly modifies the red component based on the sine of the angle, aiding
    # in hue rotation towards red.
    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]
    return np.dstack([r, g, b])


# returns the values of the array that has the highest difference between two consecutive elements
def getNormalizingInterval(array):
    high = 255
    low = 0
    max_dist = 0
    for i in range(1, len(array)):
        dist = array[i] - array[i - 1]
        if dist > max_dist:
            max_dist = dist
            high = array[i]
            low = array[i - 1]
    return low, high


def applyFilter(mat, f):
    r = mat[..., 0]
    g = mat[..., 1]
    b = mat[..., 2]

    redGain = f[0]
    redGainFromGreen = f[1]
    redGainFromBlue = f[2]
    redOffset = f[4]
    greenGain = f[6]
    greenOffset = f[9]
    blueGain = f[12]
    blueOffset = f[14]

    r = r * redGain + g * redGainFromGreen + b * redGainFromBlue + redOffset
    g = g * greenGain + greenOffset
    b = b * blueGain + blueOffset

    filtered_mat = np.dstack([r, g, b])
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)
    return filtered_mat


def getFilterValues(inputMat):
    # Resize input to a fixed 256x256 resolution since
    # finding out the filter matrix does not require high resolution
    # the smaller we do it, the less complex the calculations will be since they
    # are O(n^2) where n is the resolution of the image
    miniMat = cv2.resize(inputMat, (256, 256))
    # generate an average value of the entire matrix, and throw away the alpha channel
    matAvg = np.array(cv2.mean(miniMat)[:3], dtype=np.uint8)
    averageRed = matAvg[0]
    shiftAmount = 0
    while averageRed < MIN_AVG_RED:
        shifted = redShift(matAvg, shiftAmount)
        # not super sure why we set the combined value to the averageRed value. but this is
        # exactly how the source code did it.
        averageRed = np.mean(shifted[..., 0])
        shiftAmount += 1
        if shiftAmount > MAX_HUE_SHIFT:
            averageRed = MIN_AVG_RED
    miniMatShifted = redShift(miniMat, shiftAmount)
    # summing and setting axis=2 means we are collapsing the 3rd axis. The 3rd axis is the array of RGB values.
    # This is summing the individual RGB values and creates a grayscale image, with the shifted red
    new_r_channel = np.sum(miniMatShifted, axis=2)
    # we bound the values between 0 and 255
    new_r_channel = np.clip(new_r_channel, 0, 255)
    # use python's ellipses slicing to set the first channel to the new_r_channel
    miniMat[..., 0] = new_r_channel
    histR = cv2.calcHist([miniMat], [0], None, [256], [0, 256])
    histG = cv2.calcHist([miniMat], [1], None, [256], [0, 256])
    histB = cv2.calcHist([miniMat], [2], None, [256], [0, 256])
    normalize_mat = np.zeros((256, 3))
    width = miniMat.shape[1]
    height = miniMat.shape[0]
    threshold_level = width * height / THRESHOLD_RATIO
    for x in range(256):
        if histR[x] < threshold_level:
            normalize_mat[x][0] = x
        if histG[x] < threshold_level:
            normalize_mat[x][1] = x
        if histB[x] < threshold_level:
            normalize_mat[x][2] = x
    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255
    # not super sure what's going on , but from running it, the following low and high values tend to always be 0
    # since normalizeMat tends to have some low x values at indices x, but a huge swath of 0s, then a sudden jump to
    # some higher index value
    adjust_r_low, adjust_r_high = getNormalizingInterval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = getNormalizingInterval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = getNormalizingInterval(normalize_mat[..., 2])
    shifted = redShift(np.array([1, 1, 1]), shiftAmount)
    shifted_r, shifted_g, shifted_b = shifted[0][0]
    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)
    redOffset = -adjust_r_low * red_gain
    greenOffset = -adjust_g_low * green_gain
    blueOffset = -adjust_b_low * blue_gain
    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE
    # not super sure why this is arranged as a pseudo-matrix. Prof gpt says this is not a standard in any way.
    # these variable names are straight copied form the JS version, the correctness of the names are kinda sus
    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])


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

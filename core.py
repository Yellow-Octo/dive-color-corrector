import numpy as np
import cv2
import math

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2


def correctImage(input_path, output_path):
    raw_data = cv2.imread(input_path)  # reads image into BGR format
    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)  # converts BGR to RGB
    corrected = generatedCorrected(rgb_data)
    cv2.imwrite(output_path, corrected)


def generatedCorrected(mat):
    original_mat = mat.copy()
    filterValues = getFilterValues(mat)
    corrected = applyFilter(original_mat, filterValues)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def getFilterValues(inputMat):
    """
    Generate a color matrix via red shifts and histogram analysis.
    :param inputMat: Input matrix for which the color matrix will be calculated.
    :return: 4x4 color matrix
    ```
    [r   g   b   rGain]
    [0   g   0   gGain]
    [0   0   b   bGain]
    [0   0   0   1    ]
    ```
    """
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
    # the following is a color matrix, which is a concept that can be found in adobe flash, android, and microsoft c#
    # it should be a 5x5 matrix.
    # https://developer.android.com/reference/android/graphics/ColorMatrix.html
    # https://learn.microsoft.com/en-us/dotnet/desktop/winforms/advanced/how-to-use-a-color-matrix-to-transform-a-single-color?view=netframeworkdesktop-4.8
    #
    # The reason is that rgba is 4 components, and a 5th higher dimension is added in order
    # to add the ability to add a constant (translation).
    # the rgba vector is then also padded by 1 making it 5 components.

    # since we are not working with alpha channel, our matrix is one dimension less, but the concept is the same

    color_matrix = np.array([
        [adjust_red, adjust_red_green, adjust_red_blue, redOffset],
        [0, green_gain, 0, greenOffset],
        [0, 0, blue_gain, blueOffset],
        [0, 0, 0, 1]
    ])

    return color_matrix


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


def applyFilter(mat, color_matrix):
    """
    Apply a color matrix transformation to an image matrix
    :param mat:
    :param color_matrix: 4x4 color matrix
    :return: adjusted image
    """
    height, width = mat.shape[:2]

    # Reshape the input matrix to 2D (height * width, 3)
    reshaped_mat = mat.reshape(-1, 3)

    # Add a column of ones to create 4x1 vectors [r, g, b, 1]
    reshaped_mat_homogeneous = np.column_stack((reshaped_mat, np.ones(reshaped_mat.shape[0])))

    # Apply the color matrix transformation
    filtered_mat = np.dot(reshaped_mat_homogeneous, color_matrix.T)

    # Remove the last column (which was for the homogeneous coordinate)
    filtered_mat = filtered_mat[:, :3]

    # Reshape back to the original shape and clip values
    filtered_mat = filtered_mat.reshape(height, width, 3)
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat

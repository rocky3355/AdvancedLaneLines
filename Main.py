import numpy as np
import cv2
import glob


class CameraCalibration:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class Transformation:
    def __init__(self, matrix, matrix_inv):
        self.matrix = matrix
        self.matrix_inv = matrix_inv

    def transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.matrix, img_size, flags=cv2.INTER_LINEAR)


def print_calibration(calibration):
    # TODO: Also do for TestImages/straight_lines1.jpg
    img = cv2.imread("CameraCalibration/calibration1.jpg")
    undist = calibration.undistort_image(img)
    cv2.imwrite("test.jpg", undist)


def print_binary(img, file_name):
    img_scaled = img * 255
    img = np.dstack((img_scaled, img_scaled, img_scaled))
    cv2.imwrite(file_name, img)


def calibrate_camera():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('CameraCalibration/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    calibration = CameraCalibration(mtx, dist)
    return calibration


def color_gradient_crop(img):
    cropped = img[img.shape[0]//2:img.shape[0]]

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(cropped, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def perspective_transform():
    #src = np.float32([[360, 200], [360, 1100], [85, 680], [85, 600]])
    #dst = np.float32([[360, 200], [360, 1100], [0, 1100], [0, 200]])
    src = np.float32([[200, 360], [1100, 360], [700, 100], [580, 100]])
    dst = np.float32([[200, 360], [1100, 360], [1100, 0], [200, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    transform = Transformation(M, Minv)
    return transform

calibration = calibrate_camera()
#print_calibration(calibration)

test_img = cv2.imread('TestImages/straight_lines1.jpg')
undistorted = calibration.undistort(test_img)
combined_binary = color_gradient_crop(undistorted)
print_binary(combined_binary, 'color_and_gradient.jpg')

transform = perspective_transform()
warped = transform.transform(combined_binary)
print_binary(warped, 'warped.jpg')


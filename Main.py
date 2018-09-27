import os
import cv2
import glob
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


PRINT_STAGES = False
IMAGE_FOLDER = 'OutputImages'
TEXT_FONT = ImageFont.truetype('Fonts/Arial.ttf', 25)
TEXT_COLOR = (255, 255, 255)
LANE_WIDTH_M = 3.7
MY = 30.0 / 720.0
MX = LANE_WIDTH_M / 900.0


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

    def transform_inv(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.matrix_inv, img_size, flags=cv2.INTER_LINEAR)


def print_calibration(calibration):
    img = cv2.imread("CameraCalibration/calibration1.jpg")
    undist = calibration.undistort(img)
    cv2.imwrite(os.path.join(IMAGE_FOLDER, 'undistorted.jpg'), undist)


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


def color_gradient(img):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Brighten up the colors
    s_channel = np.multiply(s_channel, 1.8)

    if PRINT_STAGES:
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        cv2.imwrite(os.path.join(IMAGE_FOLDER, 'color.jpg'), rgb)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20 #30
    thresh_max = 100 #100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold saturation channel
    s_thresh_min = 170 #140
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Threshold hue channel
    l_thresh_min = 50
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(l_binary == 1) & ((s_binary == 1) | (sxbinary == 1))] = 1

    return combined_binary


def perspective_transform():
    # Transform the lane (trapezoid shape) to a rectangle
    src = np.float32([[200, 720], [1100, 720], [575, 450], [700, 450]])
    dst = np.float32([[200, 720], [1100, 720], [200, 0], [1100, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    transform = Transformation(M, Minv)
    return transform


def find_lane_pixels(binary_warped):
    out_img = None
    if PRINT_STAGES:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Starting points for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9
    margin = 100
    minpix = 25

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    leftx_current = leftx_base
    rightx_current = rightx_base
    lane_center = (leftx_base + rightx_base) / 2
    # Negative shift means that the car is left of the lane center
    center_shift = (midpoint - lane_center) * MX

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if PRINT_STAGES:
            # Find the peak of the left and right halves of the histogram
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, center_shift, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, center_shift, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    if PRINT_STAGES:
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        # Plots the left and right polynomials on the lane lines
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, out_img.shape[1])
        plt.ylim(out_img.shape[0], 0)
        fig.savefig(os.path.join(IMAGE_FOLDER, 'polynom.jpg'))
        plt.close(fig)

    return ploty, left_fit, right_fit, left_fitx, right_fitx, center_shift


def measure_curvature_meters(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)

    a_left = MX / (MY ** 2) * left_fit[0]
    b_left = (MX / MY) * left_fit[1]
    a_right = MX / (MY ** 2) * right_fit[0]
    b_right = (MX / MY) * right_fit[1]

    left_radius = (1 + (2 * a_left * y_eval + b_left) ** 2) ** (3 / 2) / np.abs(2 * a_left)
    right_radius = (1 + (2 * a_right * y_eval + b_right) ** 2) ** (3 / 2) / np.abs(2 * a_right)

    avg_radius = (left_radius + right_radius) / 2
    return avg_radius


def print_lane(undist, warped, ploty, left_fitx, right_fitx, transform):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = transform.transform_inv(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
    return result


def find_lane(img):
    # Undistort the image
    undist = calibration.undistort(img)

    # Filter the image by gradient and color
    combined_binary = color_gradient(undist)
    if PRINT_STAGES:
        print_binary(combined_binary, os.path.join(IMAGE_FOLDER, 'filtered.jpg'))

    # Warp the image to the bird's eye view
    transform = perspective_transform()
    warped_binary = transform.transform(combined_binary)
    if PRINT_STAGES:
        print_binary(warped_binary, os.path.join(IMAGE_FOLDER, 'warped.jpg'))

    # Find the lines
    ploty, left_fit, right_fit, left_fitx, right_fitx, center_shift = fit_polynomial(warped_binary)

    # Draw the lane onto the image
    lane_img = print_lane(undist, warped_binary, ploty, left_fitx, right_fitx, transform)

    # Calculate the lane radius
    lane_radius = measure_curvature_meters(ploty, left_fit, right_fit)
    pil_image = Image.fromarray(lane_img)
    ImageDraw.Draw(pil_image).text((200, 20), 'Curvature radius: {:.0f}m'.format(lane_radius), TEXT_COLOR, TEXT_FONT)
    ImageDraw.Draw(pil_image).text((200, 50), 'Vehicle positon: {:.2f}m {} of lane center'.format(np.abs(center_shift), 'left' if center_shift < 0 else 'right'), TEXT_COLOR, TEXT_FONT)
    lane_img = np.array(pil_image)

    return lane_img


# ----- TEST CODE -----
#calibration = calibrate_camera()
#print_calibration(calibration)
#img = cv2.imread('TestImages/test2.jpg')
#lane = find_lane(img)
#cv2.imwrite(os.path.join(IMAGE_FOLDER, 'lane.jpg'), lane)
#exit(0)


# ----- MAIN CODE -----
# Calibrate the camera
calibration = calibrate_camera()
if PRINT_STAGES:
    print_calibration(calibration)
# Load the video
input_video = VideoFileClip('Videos/project_video.mp4')
# Apply the lane finding algorithm to each frame
output_video = input_video.fl_image(find_lane)
# Save the video
output_video.write_videofile('output.mp4', audio=False)

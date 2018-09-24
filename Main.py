import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

PRINT_STAGES = True

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
    cv2.imwrite("undistorted.jpg", undist)


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

    s_channel = np.multiply(s_channel, 1.8)
    if PRINT_STAGES:
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        cv2.imwrite('color.jpg', rgb)

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
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

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(l_binary == 1) & ((s_binary == 1) | (sxbinary == 1))] = 1

    return combined_binary


def perspective_transform():
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
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 25

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
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

        # TODO: Only add the center of mass instead of all pixels?
        #       Those centers should be sufficient for polyline fitting

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

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
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
        fig.savefig('polynom.jpg')
        plt.close(fig)

    return ploty, left_fit, right_fit, left_fitx, right_fitx, out_img


def measure_curvature_pixels(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)

    a_left = left_fit[0]
    b_left = left_fit[1]
    a_right = right_fit[0]
    b_right = right_fit[1]

    left_curverad = (1 + (2 * a_left * y_eval + b_left) ** 2) ** (3 / 2) / np.abs(2 * a_left)
    right_curverad = (1 + (2 * a_right * y_eval + b_right) ** 2) ** (3 / 2) / np.abs(2 * a_right)

    return left_curverad, right_curverad


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
    img = cv2.imread('TestImages/test2.jpg')
    undist = calibration.undistort(img)
    combined_binary = color_gradient(undist)
    if PRINT_STAGES:
        print_binary(combined_binary, 'filtered.jpg')

    transform = perspective_transform()
    warped_binary = transform.transform(combined_binary)
    if PRINT_STAGES:
        print_binary(warped_binary, 'warped.jpg')

    ploty, left_fit, right_fit, left_fitx, right_fitx, line_img = fit_polynomial(warped_binary)

    #left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
    #print('Left radius: {:.0f}m'.format(left_curverad))
    #print('Right radius: {:.0f}m'.format(right_curverad))

    lane_img = print_lane(undist, warped_binary, ploty, left_fitx, right_fitx, transform)
    return lane_img


calibration = calibrate_camera()
if PRINT_STAGES:
    print_calibration(calibration)
lane_img = find_lane(None)
cv2.imwrite('lane.jpg', lane_img)
exit(0)

input_video = VideoFileClip('Videos/project_video.mp4')
output_video = input_video.fl_image(find_lane)
output_video.write_videofile('output.mp4', audio=False)

# TODO: Sliding window improvement
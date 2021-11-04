from moviepy.editor import VideoFileClip
import numpy as np
import glob
import pickle
import cv2

class HistoryKeeper(object):

    def __init__(self, window_width, window_height, margin, smooth_factor=15):
        self.recent_centers = []  # History of the centers previously found
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.smooth_factor = smooth_factor  # Size of the slice of recent centers to be considered when smoothing the window centroids

    def find_window_centroids(self, warped_image):
       
        window_centroids = []
        window = np.ones(self.window_width)
        image_width, image_height = warped_image.shape[1], warped_image.shape[0]

        window_width_center = self.window_width / 2  # This is the center of the image in the x axis
        stripe_height_boundary = int(3 * image_height / 4)
        stripe_width_boundary = int(image_width / 2)
        first_stripe_left_half = warped_image[stripe_height_boundary:, :stripe_width_boundary]
        left_sum = np.sum(first_stripe_left_half, axis=0)  # Calculates the number of pixels per column in the left half of the stripe
        left_center = np.argmax(np.convolve(window, left_sum)) - window_width_center  # Get the location and shift if to the center of the window

        first_stripe_right_half = warped_image[stripe_height_boundary:, stripe_width_boundary:]
        right_sum = np.sum(first_stripe_right_half, axis=0)  # Calculates the number of pixels per column in the right half of the stripe
        right_center = np.argmax(np.convolve(window, right_sum)) - window_width_center + stripe_width_boundary  # Get the location and shift if to the center of the window

        new_centroid = (left_center, right_center)
        window_centroids.append(new_centroid)

        # Repeat for the remaining stripes
        number_of_windows = int(image_height / self.window_height)
        for level in range(1, number_of_windows):
            stripe_height_boundary = int(image_height - (level + 1) * self.window_height)
            stripe_width_boundary = int(image_height - level * self.window_height)

            image_layer = np.sum(warped_image[stripe_height_boundary:stripe_width_boundary, :], axis=0)
            conv_signal = np.convolve(window, image_layer)

            offset = self.window_width / 2
            left_lower_bound = int(max(left_center + offset - self.margin, 0))
            left_max_bound = int(min(left_center + offset + self.margin, image_width))
            left_center = np.argmax(conv_signal[left_lower_bound:left_max_bound]) + left_lower_bound - offset

            right_lower_bound = int(max(right_center + offset - self.margin, 0))
            right_upper_bound = int(min(right_center + offset + self.margin, image_width))
            right_center = np.argmax(conv_signal[right_lower_bound:right_upper_bound]) + right_lower_bound - offset

            new_centroid = (left_center, right_center)
            window_centroids.append(new_centroid)

        self.recent_centers.append(window_centroids)

        # We take into account the last N centers to prevent wobbling and irregularities.
        most_recent_centers = self.recent_centers[-self.smooth_factor:]
        return np.mean(most_recent_centers, axis=0)

__history_keeper = HistoryKeeper(window_width=30, window_height=60, margin=25, smooth_factor=30)

def calibrate_camera(directory_path='./camera_cal', chessboard_shape=(9, 6), save_location='./cal_pickle.p'):
    """
    Takes a series of check board images and uses them to find the appropriate calibration parameters.
    :param directory_path: Directory locations where the calibration images lie.
    :param chessboard_shape: Dimensions (in squares) of the chess boards.
    :param save_location: Path of the file where the calibration parameters will be stored (pickled)
    :return:
    """
    # Let's prepare object points, like (0, 0, 0)... (8, 5, 0)
    columns, rows = chessboard_shape
    op = np.zeros((rows * columns, 3), np.float32)
    op[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    # Lists to store object points and image points
    obj_points = []  # Points in 3D, real world images.
    img_points = []  # Points in 2D images.

    image_name_pattern = directory_path + "/calibration*.jpg"
    for img_index, img_name in enumerate(glob.glob(image_name_pattern)):
        # Load the image and transform it into gray scale
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to find the corners in the image
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

        # If the corners were found, append the object and image points to result
        # and print the corners in the original image.
        if ret:
            obj_points.append(op)
            img_points.append(corners)

            cv2.drawChessboardCorners(img, chessboard_shape, corners, ret)
            cv2.imwrite(directory_path + '/corners' + str(img_index) + ".jpg", img)

    # We load the first image just to determine its dimensions.
    img = cv2.imread(directory_path + '/calibration2.jpg')
    image_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    cv2.imwrite(directory_path + '/calibration2_corrected.jpg', cv2.undistort(img, mtx, dist, None, mtx))

    if save_location:
        pickle.dump({'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}, open(save_location, 'wb'))

    return mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    calibrate_camera()

def load_calibration_parameters(location='./cal_pickle.p'):
    """
    Loads the calibration parameters (used for distortion correction) from a particular location and returns them.
    :param location: Path of the pickle file where the parameters are stored.
    :return:
    """
    with open(location, 'rb') as pickle_file:
        dist_pickle = pickle.load(pickle_file)

    return dist_pickle['mtx'], dist_pickle['dist'], dist_pickle['rvecs'], dist_pickle['tvecs']

def abs_sobel_threshold(img, orientation='x', sobel_kernel=3, threshold=(0, 255)):
    """
    Applies the sobel operation to the input image to find the gradients in a particular orientation (X or Y).
    :param img: Image whose gradients will be calculated.
    :param orientation: Orientation of the sobel. X for horizontal, Y for vertical.
    :param sobel_kernel: Kernel size. Must be an odd number greater than 3. The greater the number, the smoother the result.
    :param threshold: Pixels will be considered in the resulting mask if they are within these boundaries.
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Heads up! Change to RGB if using mpimg.imread

    if orientation.lower() == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    # Apply threshold
    lower, upper = threshold
    selector = (scaled_sobel >= lower) & (scaled_sobel <= upper)
    binary_output[selector] = 1

    return binary_output


def color_threshold(img, s_threshold=(0, 255), v_threshold=(0, 255)):
    """
    Thresholds an image using the H channel (of the HLS color space) and the V channel (of the HSV color space).
    :param img: Input image.
    :param s_threshold: Range of accepted values for the H channel.
    :param v_threshold: Range of accepted values fir the V channel.
    :return: A color mask with the same dimensions as the input image but with only one channel.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]
    s_binary = np.zeros_like(s)
    lower_s, upper_s = s_threshold
    s_binary[(s >= lower_s) & (s <= upper_s)] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_binary = np.zeros_like(v)
    lower_v, upper_v = v_threshold
    v_binary[(v >= lower_v) & (v <= upper_v)] = 1

    binary_output = np.zeros_like(s)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output


def calculate_perspective_transform_parameters():
    """
    Calculates the parameters needed to transform the perspective an image to a birds-eye view.
    :return: 1. The transformation matrix needed to pass from the original perspective to the top perspective.
             2. The inverse of the transformation matrix, which is useful when we need to revert the perspective transformation.
             3. The source points, which describe a trapezoid in the original image.
             4. The destination points, which describe a rectangle in the transformed image.
    """
    src = np.float32([[589,  446], [691,  446], [973,  677], [307,  677]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    return transform_matrix, inverse_transform_matrix, src, dst


def put_offset_and_radius(image, offset_from_center, radius):
    """
    Puts the offset from the center of the lane and the radius of the curvature on top of a provided image.
    :param image: Image where we'll place the text.
    :param offset_from_center: Offset (in meters) of the car from the center of the lane.
    :param radius: Radius (in meter) of the lane curvature.
    :return: Same image with the text on top of it.
    """
    cv2.putText(image, 'Radius of the Curvature (in meters) = ' + str(round(radius, 3)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(image, 'Car is ' + str(abs(round(offset_from_center, 3))) + ' meters off of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return image


def get_line_polygon(xs, ys, thickness=20):
    """
    Takes the X and Y values that describe the points that comprise a line, and returns the points that describe a
     polygon with the provided thickness (in pixels).
    :param xs: X coordinates.
    :param ys: Y coordinates.
    :param thickness: Width of the resulting polygon in pixels.
    :return:
    """
    all_xs = np.concatenate((xs - thickness / 2, xs[::-1] + thickness / 2), axis=0)
    all_ys = np.concatenate((ys, ys[::-1]), axis=0)

    polygon_points = np.array([(x, y) for x, y in zip(all_xs, all_ys)], np.int32)

    return polygon_points


def put_lines_on_image(input_image, reverse_perspective_transform_matrix, left_lane, right_lane, inner_lane):
    """
    Overlaps the lane lines found in a top perspective image over the original image.
    :param input_image: Original, undistorted image.
    :param reverse_perspective_transform_matrix:  Matrix used to revert the perspective transform of the lanes.
    :param left_lane: Left lane polygon points.
    :param right_lane: Right lane polygon points.
    :param inner_lane: Inner lane polygon points.
    :return: Original image with the lane lines found drawn onto it.
    """
    image_size = (input_image.shape[1], input_image.shape[0])

    base_road = np.zeros_like(input_image)
    road_background = np.zeros_like(input_image)

    cv2.fillPoly(base_road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(base_road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(base_road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_background, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_background, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(base_road, reverse_perspective_transform_matrix, image_size,
                                      flags=cv2.INTER_LINEAR)
    road_warped_background = cv2.warpPerspective(road_background, reverse_perspective_transform_matrix, image_size,
                                                 flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(input_image, 1.0, road_warped_background, -1.0, 0.0)
    return cv2.addWeighted(base, 1, road_warped, .7, 0.0)


def evaluate_polynomial(ys, coefficients):
    """
    Evaluates a 2nd degree polynomial over a collection of values.
    :param ys: Input values
    :param coefficients: Polynomial coefficients.
    :return: Output values (xs)
    """
    # Unpack coefficients
    a, b, c = coefficients[0], coefficients[1], coefficients[2]

    return np.array(a * np.power(ys, 2) + b * ys + c, np.int32)


def process_image(input_image, curve_centers=__history_keeper):
    """
    Takes an image and returns it with the lane lines, radius curvature and center offset drawn on top of it.
    :param input_image: RGB image.
    :param curve_centers: History keeper objects used to keep the centroids of the windows found in each frame.
    :return: Original image with the lane lines, radius curvature and center offset drawn on top of it.
    """
    ########################################################
    # STEP 1: Undistort image and extract image dimensions #
    ########################################################
    camera_matrix, distortion_coefficients, _, _ = load_calibration_parameters()
    input_image = cv2.undistort(input_image, camera_matrix, distortion_coefficients, None, camera_matrix)
    original_image_size = (input_image.shape[1], input_image.shape[0])
    original_image_width, original_image_height = original_image_size

    ###############################################
    # STEP 2: Apply color & gradient thresholding #
    ###############################################
    processed_image = np.zeros_like(input_image[:, :, 0])  # Just a black canvas
    x_gradient_mask = abs_sobel_threshold(input_image, orientation='x', threshold=(12, 255), sobel_kernel=3)
    y_gradient_mask = abs_sobel_threshold(input_image, orientation='y', threshold=(25, 255), sobel_kernel=3)
    color_binary_mask = color_threshold(input_image, s_threshold=(100, 255), v_threshold=(50, 255))
    selection = (x_gradient_mask == 1) & (y_gradient_mask == 1) | (color_binary_mask == 1)
    processed_image[selection] = 255

    ########################################
    # STEP 3: Apply perspective transform. #
    ########################################
    perspective_matrix, revert_perspective_matrix, src, dst = calculate_perspective_transform_parameters()
    warped = cv2.warpPerspective(processed_image, perspective_matrix, original_image_size, flags=cv2.INTER_LINEAR)

    ###########################################################
    # STEP 4: Find the centroids of each window in each line. #
    ###########################################################
    window_centroids = curve_centers.find_window_centroids(warped)

    # points used to find the lef+t and right lane lines
    right_x_points = []
    left_x_points = []

    number_of_levels = len(window_centroids)
    for level in range(number_of_levels):
        left, right = window_centroids[level]  # Current centroid
        left_x_points.append(left)
        right_x_points.append(right)

    ##########################################
    # STEP 5: Fit a polynomial to each line. #
    ##########################################
    y_values = range(200, original_image_height)
    res_y_vals = np.arange(original_image_height - (curve_centers.window_height / 2), 0, -curve_centers.window_height)

    left_polynomial_coefficients = np.polyfit(res_y_vals, left_x_points, 2)
    left_xs = evaluate_polynomial(y_values, left_polynomial_coefficients)

    right_polynomial_coefficients = np.polyfit(res_y_vals, right_x_points, 2)
    right_xs = evaluate_polynomial(y_values, right_polynomial_coefficients)

    # Left and right lines polygons
    lest_lane_vertices = get_line_polygon(left_xs, y_values, thickness=curve_centers.window_width)
    right_lane_vertices = get_line_polygon(right_xs, y_values, thickness=curve_centers.window_width)

    # The inner lane is actually the space where the car is, so it must go from the rightmost points in the left lane
    # to the leftmost points in the right lane.
    mid_point_window_height = curve_centers.window_width / 2
    inner_lane_vertices = np.array(
        list(zip(np.concatenate((left_xs + mid_point_window_height, right_xs[::-1] - mid_point_window_height), axis=0),
                 np.concatenate((y_values, y_values[::-1]), axis=0))), np.int32)

    # Put the lines on the original image.
    result_image = put_lines_on_image(input_image, revert_perspective_matrix, lest_lane_vertices, right_lane_vertices,
                                      inner_lane_vertices)

    ##############################################
    # STEP 6: Calculate radius of the curvature. #
    ##############################################
    meters_per_pixel_y_axis = 10 / 720
    meters_per_pixel_x_axis = 4 / 384
    res_y_vals_in_meters = np.array(res_y_vals, np.float32) * meters_per_pixel_y_axis
    left_x_in_meters = np.array(left_x_points, np.float32) * meters_per_pixel_x_axis
    curvature_radius_polynomial_coefficients = np.polyfit(res_y_vals_in_meters, left_x_in_meters, 2)

    # Unpack coefficients
    a, b = curvature_radius_polynomial_coefficients[0], curvature_radius_polynomial_coefficients[1]
    numerator = ((1 + (2 * a * y_values[-1] * meters_per_pixel_y_axis + b) ** 2) ** 1.5)
    denominator = np.absolute(2 * a)
    curve_radius = numerator / denominator

    #######################################################
    # STEP 7: Calculate the offset of the car on the road #
    #######################################################
    # We assume our camera is fixed at the center of the car lanes.
    lane_center = (left_xs[-1] + right_xs[-1]) / 2
    image_center = original_image_width / 2
    center_delta_in_meters = (lane_center - image_center) * meters_per_pixel_x_axis

    result_image = put_offset_and_radius(result_image, center_delta_in_meters, curve_radius)

    # Finally return the original image with all the data of interest shown.
    return result_image


def process_video(video_path, output_path='project_video_out.mp4'):
    """
    Takes an input video of lane lines and saves a new video with the lane lanes, curvature radius and camera offset from
    center annotated.
    :param video_path: Input video location.
    :param output_path: Output video location.
    """
    # Load the original video
    input_video = VideoFileClip(video_path)

    # Process and save
    processed_video = input_video.fl_image(process_image)
    processed_video.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    input_video_path = 'project_video.mp4'
    process_video(input_video_path)

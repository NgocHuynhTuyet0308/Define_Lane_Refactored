import numpy as np


class LaneDetector:
    def __init__(self):
        pass

    @staticmethod
    def fit_curve(image):
        histogram = np.sum(image, axis=0)

        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 100
        margin = 100
        minpix = 50
        window_height = int(image.shape[0] / nwindows)

        y, x = image.nonzero()
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_indices = []
        right_lane_indices = []

        for window in range(nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_indices = \
                ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
            good_right_indices = \
                ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > minpix:
                leftx_current = int(np.mean(x[good_left_indices]))
            if len(good_right_indices) > minpix:
                rightx_current = int(np.mean(x[good_right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx = x[left_lane_indices]
        lefty = y[left_lane_indices]
        rightx = x[right_lane_indices]
        righty = y[right_lane_indices]

        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = None

        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = None

        return left_fit, right_fit

    @staticmethod
    def find_points(img_shape, left_fit, right_fit):
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        pts_left = None
        pts_right = None

        if left_fit is not None:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

        if right_fit is not None:
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        if pts_left is None and pts_right is None:
            print("Lỗi: Không có fit cho làn đường trái hoặc phải.")

        return pts_left, pts_right

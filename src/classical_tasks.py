import cv2
import numpy as np


class ClassicalTasks:
    def __init__(self):
        """
        Classical CV methods: Template Matching + Optical Flow
        """
        pass

    # -------------------------
    # TEMPLATE MATCHING
    # -------------------------
    def template_match(self, image, template, method=cv2.TM_CCOEFF_NORMED):
        """
        Finds template inside image using OpenCV template matching.

        Returns:
            result_map, best_match_location
        """
        result = cv2.matchTemplate(image, template, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # best match depends on method
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_loc = min_loc
            score = min_val
        else:
            best_loc = max_loc
            score = max_val

        return result, best_loc, score

    # -------------------------
    # OPTICAL FLOW (Lucas-Kanade)
    # -------------------------
    def optical_flow(self, prev_img, next_img):
        """
        Sparse optical flow using Lucas-Kanade method.

        Returns:
            prev_points, next_points, status
        """

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        # detect good features to track
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10
        )

        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            next_gray,
            prev_points,
            None
        )

        return prev_points, next_points, status
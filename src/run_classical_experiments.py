import cv2
import os
import csv
import numpy as np
from src.classical_tasks import ClassicalTasks


class ClassicalExperimentRunner:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ct = ClassicalTasks()

        self.img_path = os.path.join(base_dir, "000000000009.jpg")
        self.template_path = os.path.join(base_dir, "000000000061.jpg")

        self.img = cv2.imread(self.img_path)
        self.template = cv2.imread(self.template_path)

        if self.img is None or self.template is None:
            raise ValueError("Images not found!")

        # -------------------------
        # 4 TRUE DISTORTIONS
        # -------------------------
        self.distortions = {
            "clean": lambda x: x,

            # 1. Gaussian noise
            "gaussian_noise": self.add_gaussian_noise,

            # 2. salt & pepper
            "salt_pepper": self.add_salt_pepper,

            # 3. low light
            "low_light": lambda x: cv2.convertScaleAbs(x, alpha=0.4, beta=0),

            # 4. motion blur (real kernel)
            "motion_blur": self.motion_blur
        }


    # -------------------------
    # TEMPLATE MATCHING BASELINE
    # -------------------------
    def run_template_matching(self):
        h, w, _ = self.img.shape

        # לקחת patch קטן כ-template
        template = self.img[100:200, 100:200]

        _, best_loc, score = self.ct.template_match(self.img, template)

        print("\n--- TEMPLATE MATCHING BASELINE ---")
        print("score:", score)
        print("location:", best_loc)
        
    
    # -------------------------
    # DISTORTION FUNCTIONS
    # -------------------------

    def add_gaussian_noise(self, img):
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def add_salt_pepper(self, img, amount=0.02):
        out = img.copy()
        h, w, c = out.shape

        num_pixels = int(amount * h * w)

        # salt (white)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        out[coords[0], coords[1]] = 255

        # pepper (black)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        out[coords[0], coords[1]] = 0

        return out

    def motion_blur(self, img):
        kernel = np.zeros((15, 15))
        kernel[7, :] = np.ones(15)
        kernel = kernel / 15
        return cv2.filter2D(img, -1, kernel)

    # -------------------------
    # TEMPLATE MATCHING
    # -------------------------
    def run_template_matching(self, output_csv="template_results.csv"):
        results = []

        for name, func in self.distortions.items():
            distorted = func(self.img)

            result_map, best_loc, score = self.ct.template_match(
                distorted,
                self.template
            )

            results.append([name, float(score), best_loc])

            print(f"[TEMPLATE] {name} -> score: {score:.4f}, loc: {best_loc}")

        self._save_csv(output_csv, results,
                       ["distortion", "score", "location"])

    # -------------------------
    # OPTICAL FLOW
    # -------------------------
    def run_optical_flow(self, output_csv="optical_flow_results.csv"):
        results = []

        img1 = self.img
        img2 = self.motion_blur(self.img)

        for name, func in self.distortions.items():
            distorted1 = func(img1)
            distorted2 = func(img2)

            prev_pts, next_pts, status = self.ct.optical_flow(
                distorted1, distorted2
            )

            good_points = int(status.sum()) if status is not None else 0

            results.append([name, good_points])

            print(f"[FLOW] {name} -> tracked points: {good_points}")

        self._save_csv(output_csv, results,
                       ["distortion", "tracked_points"])

    # -------------------------
    def _save_csv(self, path, data, header):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)


def run_all_classical():
    BASE_DIR = "/home/roni/datasets/coco128/images/train2017"

    runner = ClassicalExperimentRunner(BASE_DIR)

    print("\n--- TEMPLATE MATCHING EXPERIMENTS ---")
    runner.run_template_matching()

    print("\n--- OPTICAL FLOW EXPERIMENTS ---")
    runner.run_optical_flow()


if __name__ == "__main__":
    run_all_classical()
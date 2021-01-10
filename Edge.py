import numpy as np
from scipy.ndimage.filters import convolve
from typing import List, Tuple

# Typing vars.
GrayImg = List[List[float]]


class EdgeDetection:
    """
    Image edge detector.
    """

    _RIGHT_SOBEL_KERNEL = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    _TOP_SOBEL_KERNEL = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    def sobel_filters(self, img: GrayImg) -> Tuple[GrayImg, GrayImg]:
        """
        Convolution filter with edge detection kernels.

        Args:
            img (GrayImg): Image to convolve / diffuse on.

        Returns:
            Tuple[GrayImg, GrayImg]: Edge detection and edge angle.
        """
        Kx = np.array(self._RIGHT_SOBEL_KERNEL, np.float32)
        Ky = np.array(self._TOP_SOBEL_KERNEL, np.float32)

        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)

        # Joins x and y edge detection.
        out = np.hypot(Ix, Iy)
        out = out / out.max() * 255

        # Gets the edges angle direction.
        theta = np.arctan2(Iy, Ix)

        return out, theta

    def non_max_suppression(self, img: GrayImg, theta: GrayImg) -> GrayImg:
        """
        Removes the gaussian effect on edges.

        Args:
            img (GrayImg): Image with detected edges.
            theta (GrayImg): Edge angle.

        Returns:
            GrayImg: Detected edges with non max values supressed.
        """
        M, N = img.shape
        out = np.zeros((M, N), dtype=np.int32)

        angle = theta * 180.0 / np.pi
        angle[angle < 0] += 180  # All positives angles.

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = r = 255

                # Angle 0.
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # Angle 45.
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # Angle 90.
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # Angle 135.
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    out[i, j] = img[i, j]
                else:
                    out[i, j] = 0

        return out

    def threshold(
        self, img: GrayImg, lowThresholdRatio=0.05, highThresholdRatio=0.1
    ) -> GrayImg:
        """
        Strongs sharpened edges.

        Args:
            img (GrayImg): Image with edges.
            lowThresholdRatio (float, optional): Low thresold. Defaults to 0.05.
            highThresholdRatio (float, optional): High threshold. Defaults to 0.09.

        Returns:
            GrayImg: Stronged image.
        """
        M, N = img.shape
        out = np.zeros((M, N), dtype=np.int32)

        # Normalize threscholds.
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        # Weak and strong pixel values.
        weak = np.int32(25)
        strong = np.int32(255)

        # Get "strong" image positions.
        strong_i, strong_j = np.where(img >= highThreshold)
        # Get "weak" image positions.
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        # All strong pixels to strong value.
        out[strong_i, strong_j] = strong
        # All weak pixels to weak value.
        out[weak_i, weak_j] = weak

        return out, weak, strong

    def hysteresis(self, img: GrayImg, weak=25, strong=255) -> GrayImg:
        """
        Removes all weak edges if no strong adjacent pixel is detected.

        Args:
            img (GrayImg): Thresholded image with weak and strong only values.
            weak (int, optional): Weak value to remove. Defaults to 25.
            strong (int, optional): Strong value. Defaults to 255.

        Returns:
            GrayImg: Cleaned image with strong only value edges.
        """
        M, N = img.shape
        out = np.copy(img)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    # If adjacent strong pixel -> Change to strong.
                    if (
                        any(
                            [
                                img[i + 1, j - 1],
                                img[i + 1, j],
                                img[i + 1, j + 1],
                                img[i, j - 1],
                                img[i, j + 1],
                                img[i - 1, j - 1],
                                img[i - 1, j],
                                img[i - 1, j + 1],
                            ]
                        )
                        == strong
                    ):
                        out[i, j] = strong
                    # If no adjacent strong pixel -> Remove.
                    else:
                        out[i, j] = 0

        return out

    def detect_edges(self, img: GrayImg) -> GrayImg:
        """
        Edge detection workflow.

        Args:
            img (GrayImg): Input image. (Must be denoised first).

        Returns:
            GrayImg: Output image with detected edges.
        """
        # Edge and angle.
        soft_edge_detection, edge_dir = self.sobel_filters(img)

        # Gaussian effect removal.
        thin_edges_detection = self.non_max_suppression(soft_edge_detection, edge_dir)

        # Thresholding image.
        thresholded_edge_detection, weak_edges, strong_edges = self.threshold(
            thin_edges_detection
        )

        # Weak pixels removal.
        clean_edge_detection = self.hysteresis(
            thresholded_edge_detection, weak_edges, strong_edges
        )

        return clean_edge_detection

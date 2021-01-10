import numpy as np
from random import random
from typing import List, Tuple

# Typing.
GrayImg = Neighs = Kernel = List[List[float]]
RGBImg = List[List[List[float]]]
Shape = Tuple[int, int]


class Noiser:
    """
    Image noiser. Adds noise to a given image.
    """

    def rgb2gray(self, rgb_img: RGBImg) -> GrayImg:
        """
        Convert colored images to grayscale.

        Args:
            rgb_img (RGBImg): RGB image.

        Returns:
            GrayImg: Grayscale image.
        """
        rgb_weights = [0.2989, 0.5870, 0.1140]
        gray_img = np.dot(rgb_img, rgb_weights)
        return gray_img

    def create_sp_noise(self, img: GrayImg, prob=0.1) -> GrayImg:
        """
        Adds Salt & Pepper noise to an image.

        Args:
            img (GrayImg): Gray image to add noise to.
            prob (float, optional): Amount of noise. Defaults to 0.1.

        Returns:
            GrayImg: Output image.
        """
        prob /= 2
        output = np.zeros(img.shape, np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    def __gaussian_noise(self, img: GrayImg, sigma=0.1) -> GrayImg:
        """
        Creates gaussian noise overlay.

        Args:
            img (GrayImg): Grayscale image.
            sigma (float, optional): Amount of noise.

        Returns:
            GrayImg: Gaussian overlay.
        """
        mu = 0.0
        output = np.random.normal(mu, sigma, img.shape)
        return output

    def create_gaussian_noise(self, img: GrayImg, sigma=0.1) -> GrayImg:
        """
        Adds gaussian noise to an image.

        Args:
            img (GrayImg): Image to add noise to.
            sigma (float, optional): Amount of noise. Defaults to 0.15.

        Returns:
            GrayImg: Image with gaussian noise.
        """
        sigma *= 100
        gaussian = self.__gaussian_noise(img, sigma)
        out = img + gaussian
        return out

    def create_speckle_noise(self, img: GrayImg, sigma=0.1) -> GrayImg:
        """
        Adds speckle noise to an image.

        Args:
            img (GrayImg): Image to add noise to.
            sigma (float, optional): Amount of noise. Defaults to 0.15.

        Returns:
            GrayImg: Image with speckle noise.
        """
        gaussian = self.__gaussian_noise(img, sigma)
        out = img + img * gaussian
        return out


class Denoiser:
    """
    Denoiser with various image filters.
    """

    def __threshold(self, th=0.1) -> Tuple[float, float]:
        """
        Returns the double threshold between 0-255.

        Args:
            th (float, optional): threshold between 0-1. Defaults to 0.1.

        Returns:
            Tuple[float, float]: Double threshold.
        """
        # Threshold is range 0-1.
        th = th / 2 * 255
        # Because we apply it on both sides.
        th = 0 + th, 255 - th
        return th

    def __check_bounds(self, x: int, d: int, shape: int) -> Tuple[int, int, Shape]:
        """
        Checks the bounds of a matrix within a given shape.

        Args:
            x (int): Initial value.
            d (int): Dimension from initial value.
            shape (int): Max value.

        Returns:
            Tuple[int, int, Shape]: New matrix boundaries and padding if needed.
        """
        n = d // 2

        # Calculate bounds.
        b1 = x - n
        b2 = x + n + 1

        # Get bounded coordinates.
        x1 = min(max(b1, 0), shape)
        x2 = max(min(b2, shape), 0)

        # Set padding if bound out of image or 0 otherwise.
        p1 = abs(b1) if b1 < 0 else 0
        p2 = b2 - shape if b2 > shape else 0
        padding = (p1, p2)

        return x1, x2, padding

    def __get_neighs(self, img: GrayImg, x: int, y: int, d: int, pad=False) -> Neighs:
        """
        Gets the neighbors of a certain pixel of an image.

        Args:
            img (GrayImg): Image to get the neighbors of.
            x (int): Number of row.
            y (int): Number of column.
            d (int): Size of neighs matrix.
            pad (bool, optional): Used to maintain neighs matrix size on borders. Defaults to False.

        Returns:
            Neighs: Neighbors of pixel.
        """
        # Get new bounds and padding.
        x1, x2, padding_x = self.__check_bounds(x, d, img.shape[1])
        y1, y2, padding_y = self.__check_bounds(y, d, img.shape[0])

        # Create neighs.
        neighs = img[y1:y2, x1:x2]

        if pad:
            # Padding given the x, y values.
            padding = [padding_y, padding_x]
            # Pad neighs.
            val = img[y][x]
            neighs = np.pad(
                neighs, padding, mode="constant", constant_values=(val, val)
            )

        return neighs

    def __manual_median_filter(self, neighs: Neighs) -> float:
        """
        Calculates the median value of a kernel.

        Args:
            neighs (Neighs): Kernel to calculate median value.

        Returns:
            float: Median value.
        """
        neighs = np.array(neighs)
        f_neighs = neighs.flatten()
        l_neighs = len(f_neighs)
        new_pix = f_neighs[l_neighs // 2]
        return new_pix

    def __median_filter(self, neighs: Neighs) -> float:
        """
        Calculates the median value of passed kernel.
        Using numpy library.

        Args:
            neighs (Neighs): Kernel to calculate median value.

        Returns:
            float: Median value.
        """
        return np.median(neighs)

    def __manual_mean_filter(self, neighs: Neighs) -> float:
        """
        Calculates the mean value of passed kernel.

        Args:
            neighs (Neighs): Kernel to calculate mean value.

        Returns:
            float: Mean value.
        """
        neighs = np.array(neighs)
        f_neighs = neighs.flatten()
        l_neighs = len(f_neighs)
        new_pix = np.sum(f_neighs) / l_neighs
        return new_pix

    def __mean_filter(self, neighs: Neighs) -> float:
        """
        Calculates the mean value of passed kernel.
        Using numpy library.

        Args:
            neighs (Neighs): Kernel to calculate mean value.

        Returns:
            float: Mean value.
        """
        return np.mean(neighs)

    def __gaussian_filter(self, neighs: Neighs, sigma=10) -> float:
        """
        Creates new pixel value given the pixel neighbors and gaussian kernel.

        Args:
            neighs (Neighs): Pixel neighbors.
            sigma (int, optional): Amount of gaussian filtering. Defaults to 10.

        Returns:
            float: New pixel value.
        """

        def gaussian_kernel(n=5, sigma=10) -> Kernel:
            """
            Returns a gaussian kernel with given dimension.

            Args:
                n (int, optional): Dimension of kernel. Defaults to 5.
                sigma (int, optional): Amount of gaussian filtering. Defaults to 10.

            Returns:
                Kernel: Gaussian kernel.
            """
            dim = (n - 1) // 2
            k = np.linspace(-dim, dim, n)
            x, y = np.meshgrid(k, k)
            kernel = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
            return kernel

        kernel = gaussian_kernel(neighs.shape[0], sigma)
        f_kernel, f_neighs = kernel.flatten(), neighs.flatten()
        new_pix = np.dot(f_neighs, f_kernel) / np.sum(f_kernel)
        return new_pix

    def __anisotropic_filter(self, old_pix: float, neighs: Neighs, K=10):
        """
        Pixel anisotropic transformation.

        Args:
            old_pix (float): Old pixel.
            neighs (Neighs): Old pixel neighbors.
            K (int, optional): Amount of needed similarity. Defaults to 10.
        """

        def similarity_kernel(pix: float, neighs: Neighs, K=10) -> Kernel:
            """
            Gets the pixel similarity kernel.

            Args:
                pix (float): Pixel to get the similarity kernel of.
                neighs (Neighs): Pixel Neighbors.
                K (int, optional): Amount of needed similarity. Defaults to 10.

            Returns:
                Kernel: [description]
            """

            def similarity_gray(p: float, q: float, K=10) -> float:
                """
                Gets the similarity value of two pixels.

                Args:
                    p (float): 1st Pixel.
                    q (float): 2nd Pixel.
                    K (int, optional): Amount of needed similarity. Defaults to 10.

                Returns:
                    float: Similarity value.
                """
                dif = abs(float(p) - float(q))
                sim = 1.0 if dif == 0.0 else 1.0 / (1.0 + (dif / K) ** 2)
                return sim

            kernel = [similarity_gray(pix, n, K) for n in neighs]
            return kernel

        neighs = neighs.flatten()
        kernel = similarity_kernel(old_pix, neighs, K)
        new_pix = np.dot(neighs, kernel) / np.sum(kernel)
        return new_pix

    def __diffusion(
        self, img: GrayImg, filter: str, k_size=8, th=1, sigma=10, K=200
    ) -> GrayImg:
        """
        Convolution of image with different denoising filters.

        Args:
            img (GrayImg): Img to filter.
            filter (str): Filter type.
            k_size (int, optional): Size of kernel for neighbors and other kernel. Defaults to 8.
            th (int, optional): Threshold for pixel values. Defaults to 1.
            sigma (int, optional): Amount of gaussian filtering. Defaults to 10.
            K (int, optional): Amount of similarity needed in anisotropic filter. Defaults to 200.

        Returns:
            GrayImg: Filtered image.
        """
        h, w = img.shape
        b_th, w_th = self.__threshold(th)

        out = np.copy(img)

        for x in range(w):
            for y in range(h):

                pix = out[y][x]
                if not (b_th < pix < w_th):
                    if filter == "gaussian":
                        neighs = self.__get_neighs(img, x, y, k_size, pad=True)
                        new_pix = self.__gaussian_filter(neighs, sigma)
                    else:
                        neighs = self.__get_neighs(img, x, y, k_size)
                        if filter == "median":
                            new_pix = self.__median_filter(neighs)
                        elif filter == "mean":
                            new_pix = self.__mean_filter(neighs)
                        elif filter == "anisotropic":
                            new_pix = self.__anisotropic_filter(pix, neighs, K)

                    out[y][x] = new_pix

        return out

    def gaus_diffusion(self, img: GrayImg, k_size=8, th=1, sigma=10) -> GrayImg:
        """
        Gaussian image filtering.

        Args:
            img (GrayImg): Image to denoise.
            k_size (int, optional): Neigs and kernel size. Defaults to 8.
            th (int, optional): Pixel threshold. Defaults to 1.
            sigma (int, optional): Amount of gaussian filtering. Defaults to 10.

        Returns:
            GrayImg: Filtered image.
        """
        return self.__diffusion(img, "gaussian", k_size, th, sigma)

    def median_diffusion(self, img: GrayImg, k_size=6, th=1) -> GrayImg:
        """
        Median image filtering.

        Args:
            img (GrayImg): Image to denoise.
            k_size (int, optional): Neigs and kernel size. Defaults to 6.
            th (int, optional): Pixel threshold. Defaults to 1.

        Returns:
            GrayImg: Filtered image.
        """
        return self.__diffusion(img, "median", k_size, th)

    def mean_diffusion(self, img: GrayImg, k_size=6, th=1) -> GrayImg:
        """
        Mean image filtering.

        Args:
            img (GrayImg): Image to denoise.
            k_size (int, optional): Neigs and kernel size. Defaults to 6.
            th (int, optional): Pixel threshold. Defaults to 1.

        Returns:
            GrayImg: Filtered image.
        """
        return self.__diffusion(img, "mean", k_size, th)

    def anis_diffusion(self, img: GrayImg, k_size=8, th=1, K=200) -> GrayImg:
        """
        Anisotropic image filtering.

        Args:
            img (GrayImg): Image to denoise.
            k_size (int, optional): Neigs and kernel size. Defaults to 6.
            th (int, optional): Pixel threshold. Defaults to 1.
            K (int, optional): Amount of pixel similarity. Defaults to 200.

        Returns:
            GrayImg: Filtered image.
        """
        return self.__diffusion(img, "anisotropic", k_size, th, K=K)

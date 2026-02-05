import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self, k=18):
        """
        Feel free to add any number of parameters here.
        But be sure to set default values. Those will be used on the evaluation server
        """
        self.dtype = np.float16
        self.mean_vector = None
        self.k = k
        self.U = np.array([], dtype=self.dtype)

    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        codebook = np.vstack((self.U, self.mean_vector))
        return codebook.astype(self.dtype)

    def train(self, train_images):
        """
        Training phase of your algorithm - e.g. here you can perform PCA on training data

        Args:
            train_images  ... A list of NumPy arrays.
                              Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
        """
        X = np.array([img.flatten() for img in train_images], dtype = np.float32)
        # center the data
        self.mean_vector = np.mean(X, axis=0)
        print(self.mean_vector.shape)
        X_centered = X - self.mean_vector

        # SVD on centered data
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Take top k principal components (right singular vectors)
        self.U = Vt[:self.k, :].astype(self.dtype)  # shape: (k, D)

    def compress(self, test_image):
        """ Given an array of shape H x W x C return compressed code """
        test_image = test_image.flatten().astype(np.float32)
        test_image_centered = test_image - self.mean_vector
        test_code = np.dot(self.U, test_image_centered)
        return test_code.astype(self.dtype)


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook.astype(np.float32)
        self.mean_vector = codebook[-1, :]
        self.U = codebook[:-1, :]

    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """
        test_image = np.dot(test_code, self.U) + self.mean_vector
        test_image = np.clip(test_image, 0, 255)
        return test_image.reshape(96, 96, 3).astype(np.uint8)
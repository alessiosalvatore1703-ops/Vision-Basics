import numpy as np
import math as math
from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches


class ImageSegmenter:
    def __init__(self,k_fg = 25, k_bg = 25, weight = 13,  mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode= mode
        self.k_fg = k_fg  
        self.k_bg = k_bg  
        self.spatial_weight = 8

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
        
    def extract_features_(self, sample_dd):
        """ Extract features from the RGB image """
        img = sample_dd['img']
        scribble_fg = sample_dd['scribble_fg']
        scribble_bg = sample_dd['scribble_bg']
        mask_fg = (scribble_fg == 255)
        mask_bg = (scribble_bg == 255)
        H, W, C = sample_dd['img'].shape
        colors_fg = img[mask_fg]
        colors_fg_norm = colors_fg.astype(float) / 255.0

        # 2. Coordinate (x, y) e normalizzazione/ponderazione
        coords_fg_y, coords_fg_x = np.where(mask_fg)
        coords_fg_x_norm_w = (coords_fg_x.astype(float) / W) * self.spatial_weight
        coords_fg_y_norm_w = (coords_fg_y.astype(float) / H) * self.spatial_weight

        # 3. Combina in un array (N_fg, 5)
        data_fg = np.column_stack((
            colors_fg_norm,
            coords_fg_x_norm_w,
            coords_fg_y_norm_w
        ))

        # --- Feature Background ---
        colors_bg = img[mask_bg]
        colors_bg_norm = colors_bg.astype(float) / 255.0

        # 2. Coordinate (x, y) e normalizzazione/ponderazione
        coords_bg_y, coords_bg_x = np.where(mask_bg)
        coords_bg_x_norm_w = (coords_bg_x.astype(float) / W) * self.spatial_weight
        coords_bg_y_norm_w = (coords_bg_y.astype(float) / H) * self.spatial_weight
        
        # 3. Combina in un array (N_bg, 5)
        data_bg = np.column_stack((
            colors_bg_norm,
            coords_bg_x_norm_w,
            coords_bg_y_norm_w
        ))

        return img, data_fg, data_bg
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        img, data_fg, data_bg = self.extract_features_(sample_dd)
        #make prediction given the k_means
        #i need to return a np matrix of bool, containing the mask
        #consider each pixel as a test point and predict if the class is bck ground or foreground
        #compute centroids fg
        centroids_fg = kmeans_fit(data_fg, self.k_fg , self.rng)
        #compute centroids bg
        centroids_bg = kmeans_fit(data_bg, self.k_bg , self.rng)
        #now i have to make predictions, how is done?
        #look for the centroid that is closer to my pixel
        cluster = np.vstack([centroids_bg, centroids_fg])  
        label_train = np.concatenate([
        np.zeros(self.k_bg, dtype=int),
        np.ones(self.k_fg, dtype=int)
        ])
        colors_test_norm = (img.astype(float) / 255.0).reshape(-1, C)
        rows, cols = np.indices((H, W))  # rows = y, cols = x
        coords_test_x = cols.reshape(-1)
        coords_test_y = rows.reshape(-1)

        coords_test_x_norm_w = (coords_test_x.astype(float) / W) * self.spatial_weight
        coords_test_y_norm_w = (coords_test_y.astype(float) / H) * self.spatial_weight
        img_reshaped_5d = np.column_stack((
            colors_test_norm,
            coords_test_x_norm_w,
            coords_test_y_norm_w
        ))

        # 5. Classifica tutti i pixel usando 1-NN sui centroidi 5D
        mask = kNN(cluster, label_train, img_reshaped_5d, k=1)
        
        # 6. Riforma la maschera finale
        mask = mask.reshape(H, W)
        return mask
    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

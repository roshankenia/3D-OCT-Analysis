import os
import glob
import cv2
import numpy as np
from abc import abstractmethod, ABCMeta
from scipy.ndimage import rotate
from functools import reduce
import torch
import random
from scipy.ndimage import zoom


class Transform(metaclass=ABCMeta):

    def __init__(self, random=False, **kwargs):
        # Should handle 3D images with or without sample and channel dimensions
        # i.e. dims = (N, C, Z, Y, X) where N and C dimensions are optional
        self.valid_dims = set([3, 4, 5])
        self.set_params(random=random, **kwargs)
        self.procedure = self.get_procedure()

    def apply(self, image, reverse=False):
        '''
        Apply transform to image. Can be modified in subclass to only
        transform parts of image (e.g. spatial dimensions).
        '''
        return self._apply(image.copy(), reverse=reverse)

    def _apply(self, image, reverse=False):
        ''' Compose and apply transformation functions '''
        assert image.ndim in self.valid_dims
        procedure = reversed(self.procedure) if reverse else self.procedure
        _transform = reduce(self._compose, self.procedure)
        return _transform(image, reverse)

    @staticmethod
    def _compose(f, g):
        return lambda im, rev: g(f(im, rev), rev)

    @abstractmethod
    def set_params(self, random=False, **kwargs):
        ''' Set params for this transform '''

    @abstractmethod
    def get_procedure(self):
        ''' Return list of functions that will applied to image sequentially '''

    @abstractmethod
    def is_spatial():
        ''' True if transformation is purely spatial '''


class GrayscaleAugmentation(Transform):
    ''' Alter brightness and contrast. '''

    def set_params(self, random=False, contrast=0.5, brightness=0.5, luminance=0.5):
        if random:
            rand = torch.rand(3).numpy()
            contrast = rand[0]
            brightness = rand[1]
            luminance = rand[2]
        assert contrast > 0 and contrast <= 1
        assert brightness > 0 and brightness <= 1
        assert luminance > 0 and luminance <= 1
        self.alpha = 1 + (contrast - 0.5) * 0.3
        self.beta = (brightness / 2) * 0.3  # only positive adjustments
        self.gamma = 2.0**(luminance * 2 - 1)

    def apply(self, image, reverse=False):
        ''' Apply each function to grayscale channel '''
        # If there are multiple channels, only treat first as grayscale
        image = image.copy()
        if image.ndim > 3:
            image[..., 0, :, :, :] = self._apply(
                image[..., 0, :, :, :], reverse=reverse)
        else:
            image = self._apply(image, reverse=reverse)

        return image

    def get_procedure(self):
        return [self.adjust_contrast, self.adjust_brightness, self.adjust_luminance]

    def adjust_contrast(self, image, reverse=False):
        if reverse:
            return image / self.alpha
        return image * self.alpha

    def adjust_brightness(self, image, reverse=False):
        ''' Note: due to clipping, this function is not fully reversible '''
        if reverse:
            return np.clip(image - self.beta, 0, 1)
        return np.clip(image + self.beta, 0, 1)

    def adjust_luminance(self, image, reverse=False):
        if reverse:
            return image**(1/self.gamma)
        return image**self.gamma

    @staticmethod
    def is_spatial():
        return False


class AffineTransformation(Transform):
    ''' Rotation and flipping. '''

    def set_params(self, random=False, angle=0, flip=[0, 0, 0],
                   rotation_step=1, interpolation=0):
        self.interpolation = interpolation
        if random:
            self.flip = torch.rand(3).numpy() > 0.5
            self.angle = torch.randint(
                0, 360//rotation_step + 1, (1,)).item()*rotation_step
        else:
            self.flip = flip
            self.angle = angle

    def get_procedure(self):
        return [self.apply_flip, self.apply_rotate]

    def apply_flip(self, image, reverse=False):
        first_spatial_dim = image.ndim - 3
        # Random flipping about each axis
        for axis, flip in enumerate(self.flip):
            image = np.flip(image, axis=axis + first_spatial_dim)
        return image

    def apply_rotate(self, image, reverse=False):
        '''
        Note: rotations are not reversible if angle is not a multiple of
        90 degrees or the dimensions along rotated axes are not equal
        '''
        first_spatial_dim = image.ndim - 3
        # Random X-Y rotation with reflective padding
        axes = (first_spatial_dim + 2, first_spatial_dim + 1)
        angle = self.angle
        if reverse:
            angle *= -1
        return rotate(image, angle, axes=axes, order=self.interpolation,
                      reshape=False, mode='reflect')

    @staticmethod
    def is_spatial():
        return True


class ZoomTransformation(Transform):
    ''' Random zoom on volume '''

    def set_params(self, random='True', range=(1, 2)):
        self.range = range

    def get_procedure(self):
        return [self.apply_zoom]

    def apply_zoom(self, image, reverse=False):

        orig_shape = image.shape

        # randomly sample float within range
        zoom_factor = np.random.uniform(self.range[0], self.range[1])

        if zoom_factor < 1.02:
            return image
        # apply zoom
        image = zoom(image, (zoom_factor, zoom_factor, zoom_factor), order=0)

        # resample to original size
        z_ind = np.random.randint(0, image.shape[0] - orig_shape[0])
        x_ind = np.random.randint(0, image.shape[1] - orig_shape[1])
        y_ind = np.random.randint(0, image.shape[2] - orig_shape[2])

        image = image[z_ind:z_ind+orig_shape[0], x_ind:x_ind +
                      orig_shape[1], y_ind:y_ind+orig_shape[2]]

        return image

    @staticmethod
    def is_spatial():
        return True

class FlipTransformation(Transform):
    ''' Random flip on volume '''

    def set_params(self, random='True'):
        self.do_flip = np.random.rand() > 0.5

    def get_procedure(self):
        return [self.apply_flip]

    def apply_flip(self, image, reverse=False):

        if not self.do_flip:
            return image

        #flip volume along x-axis
        image = np.flip(image, axis=-1)
        return image

    @staticmethod
    def is_spatial():
        return True

class FlipZTransformation(Transform):
    ''' Random flip on sup-inf axis of volume '''

    def set_params(self, random='True'):
        self.do_flip = np.random.rand() > 0.5

    def get_procedure(self):
        return [self.apply_flip]

    def apply_flip(self, image, reverse=False):

        if not self.do_flip:
            return image

        #flip volume along y-axis
        image = np.flip(image, axis=-3)
        return image

    @staticmethod
    def is_spatial():
        return True

from PIL import Image
transform_dir = './example_flip_transforms/'
if not os.path.isdir(transform_dir):
    os.makedirs(transform_dir)


glaucoma_dir = glob.glob(
    "../../CenteredData/3D_Data/Numpy_Topcon_3D_Volumes/Glaucomas" + '/**/*.npy', recursive=True)
for i in range(2):
    # zoomer = ZoomTransformation(range=(1, 1.5))
    # grayer = GrayscaleAugmentation()
    flipper = FlipTransformation()
    zflipper = FlipZTransformation()
    vol = np.load(glaucoma_dir[i])

    # resize uniform
    new_vol = np.zeros([128, 192, 112], dtype=np.float32)
    for slice in range(len(vol)):
        resized = cv2.resize(vol[slice], (112, 192))
        new_vol[slice] = resized
    vol = new_vol

    flipped = flipper.apply(vol)
    z_flipped = zflipper.apply(vol)
    both_flipped = zflipper.apply(vol)

    imgs = [Image.fromarray(img) for img in new_vol]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(transform_dir, f"og.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)
        
    imgs = [Image.fromarray(img) for img in flipped]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(transform_dir, f"viz_x_flip.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)
    
    imgs = [Image.fromarray(img) for img in z_flipped]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(transform_dir, f"viz_z_flip.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)

    imgs = [Image.fromarray(img) for img in both_flipped]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(transform_dir, f"viz_both_flip.gif"),
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)

    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og.png'), vol[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_zoomed.png'), zoomed_in[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_gray.png'), gray[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_flipped.png'), flipped[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped.png'), z_flipped[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og_32.png'), vol[32])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped_32.png'), z_flipped[32])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og_96.png'), vol[96])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped_96.png'), z_flipped[96])

    # rescale = np.max(vol) > 1
    # if rescale:
    #     vol = vol/255

    # zoomed_in = zoomer.apply(vol)
    # gray = grayer.apply(vol)
    # flipped = flipper.apply(vol)
    # z_flipped = zflipper.apply(vol)

    # if rescale:
    #     zoomed_in = (zoomed_in*255).astype(int)
    #     gray = (gray*255).astype(int)
    #     flipped = (flipped*255).astype(int)
    #     vol = (vol*255).astype(int)
    #     z_flipped = (z_flipped*255).astype(int)
        

    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og.png'), vol[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_zoomed.png'), zoomed_in[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_gray.png'), gray[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_flipped.png'), flipped[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped.png'), z_flipped[64])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og_32.png'), vol[32])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped_32.png'), z_flipped[32])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_og_96.png'), vol[96])
    # cv2.imwrite(os.path.join(transform_dir, f'{i}_z_flipped_96.png'), z_flipped[96])

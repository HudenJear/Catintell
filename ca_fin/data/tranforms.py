import cv2,math
import random
import torch
import torchvision
from torchvision.utils import make_grid
import numpy as np

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs




def paired_augment(hq_imgs,lq_imgs, resize= True, fine_size=1280, flip=True,crop=False,crop_size=384):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    flip_possibility=random.randint(0,9)%2


    def _augment(img):
        transforms = torchvision.transforms.Compose([
                    # torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                    #                                  std=(0.229, 0.224, 0.225)),
                                                     ])
        
        if resize:
          resi=torchvision.transforms.Resize(( fine_size, fine_size))
          img=resi(img)
        if flip:
          fli=torchvision.transforms.RandomHorizontalFlip(p=flip_possibility)
          img=fli(img)
          # lq_img=fli(lq_img)
        img=transforms(img)
        # lq_img=transforms(lq_img)
        return img#,lq_img

    

    if not isinstance(hq_imgs, list):
      hq_imgs = [hq_imgs]
    if not isinstance(lq_imgs, list):
      lq_imgs = [lq_imgs]
    hq_imgs = [_augment(hq_img) for hq_img in hq_imgs]
    if len(hq_imgs) == 1:
      hq_imgs = hq_imgs[0]
    lq_imgs = [_augment(lq_img) for lq_img in lq_imgs]
    if len(lq_imgs) == 1:
      lq_imgs = lq_imgs[0]
    
    if crop:
      hq_imgs,lq_imgs=paired_random_crop(hq_imgs,lq_imgs,crop_size)
    return hq_imgs,lq_imgs

def paired_random_crop(img_hq, img_lq, gt_patch_size, scale=1):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_hq, list):
        img_hq = [img_hq]
    if not isinstance(img_lq, list):
        img_lq = [img_lq]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_hq[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lq[0].size()[-2:]
        h_hq, w_hq = img_hq[0].size()[-2:]
    else:
        h_lq, w_lq = img_lq[0].shape[0:2]
        h_hq, w_hq = img_hq[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_hq != h_lq * scale or w_hq != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_hq}, {w_hq}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        # img_lq = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lq]
        img_lq = [v[:, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lq]
    else:
        img_lq = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq]

    # crop corresponding gt patch
    top_hq, left_hq = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        # img_hq = [v[:, :, top_hq:top_hq + gt_patch_size, left_hq:left_hq + gt_patch_size] for v in img_hq]
        img_hq = [v[:, top_hq:top_hq + gt_patch_size, left_hq:left_hq + gt_patch_size] for v in img_hq]
    else:
        img_hq = [v[top_hq:top_hq + gt_patch_size, left_hq:left_hq + gt_patch_size, ...] for v in img_hq]
    if len(img_hq) == 1:
        img_hq = img_hq[0]
    if len(img_lq) == 1:
        img_lq = img_lq[0]
    return img_hq, img_lq

def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def center_crop(img):
    """Cut the image into square before resize.
    If image is not square `cause of the black edge cutting, it would effect the resize result

        Args:
            img (ndarray): Image to be rotated. (loaded by cv2)
    """
    (h, w) = img.shape[:2]
    # center = (int(w // 2), int(h // 2))
    if h>w:
        cut_lenth = (h - w)//2
        fine_img=img[cut_lenth:h-cut_lenth,0:w]
    else:
        cut_lenth = (w-h) // 2
        fine_img=img[0:h,cut_lenth:w-cut_lenth]

    return  fine_img

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def single_augment(imgs, resize= True, fine_size=1280, flip=True,crop=False,crop_size=384):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """

    def _augment(img):
        transforms =torchvision.transforms.ToTensor()
        if resize:
          resi=torchvision.transforms.Resize(( fine_size, fine_size))
          img=resi(img)
        if flip:
          fli=torchvision.transforms.RandomHorizontalFlip()
          img=fli(img)
        if crop:
            cro=torchvision.transforms.RandomCrop(size=crop_size)
            img=cro(img)
        img=transforms(img)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    return imgs


def ratio_resize(imgs,min_multiplier,long_edge):
    '''
    Keep the crop ratio and resize the image into a set edge lenth. Minimum multiplier will make the image suitable for restoration network.
    imgs: Tensors

    '''
    def _r_resize(sub_img):
        h,w=sub_img.size
        if h>w:
            fine_h=long_edge
            fine_w=min_multiplier*round(long_edge*w/h/min_multiplier)
            resi=torchvision.transforms.Resize(( fine_w, fine_h))
            return resi(sub_img)
        else:
            fine_w=long_edge
            fine_h=min_multiplier*round(long_edge*h/w/min_multiplier)
            resi=torchvision.transforms.Resize(( fine_w, fine_h))
            return resi(sub_img)

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_r_resize(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    return imgs
    
    


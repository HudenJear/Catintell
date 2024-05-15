import pandas as pd
import numpy as np
import os,cv2,random,glob
from torch.utils import data as data
from .tranforms import ratio_resize,paired_augment,single_augment
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from torchvision.transforms.functional import to_tensor


class ImgTestDataset(data.Dataset):
    """Paired image dataset for image generation.

    Read image and its image pairs.

    There is 1 mode:
    single images with a individual name + image pair.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their class.
            class (int/float): the classification label of the image.
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
      super(ImgTestDataset, self).__init__()
      self.opt = opt
      # file client (io backend)
      self.mean = opt['mean'] if 'mean' in opt else None
      self.std = opt['std'] if 'std' in opt else None
      self.resize = opt['resize'] if 'reszie' in opt else True
      self.crop = opt['crop'] if 'crop' in opt else False
      self.random_crop=opt['rand_crop'] if 'rand_crop' in opt else False
      self.center_crop_size = opt['center_crop_size'] if 'center_crop_size' in opt else None
      if 'min_multiplier' in opt:
        self.resize = False
        self.min_multiplier=opt['min_multiplier'] 
      else: 
          self.min_multiplier=1

      self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

      self.path_list=[]
      for suf in ['.png','.jpg','.jpeg','.PNG']:
          sub_list=glob.glob(os.path.join(opt['image_folder'],"*"+suf))
          self.path_list.extend(sub_list)
      
      self.data_list=[]
      for indx in range(len(self.path_list)):
          lq_im=Image.open(self.path_list[indx])
          if self.min_multiplier!=1:
              # print('ratio resize activated!!')
              # print(self.resize)
              self.data_list.append(ratio_resize(lq_im,self.min_multiplier,opt['fine_size']))
          else:
              self.data_list.append([lq_im])
          

    def __getitem__(self, index):
        
        img_data=single_augment(self.data_list[index].convert('RGB'),
                                flip=self.opt['flip'], 
                                fine_size=self.opt['fine_size'],
                                crop=self.crop,
                                crop_size=self.opt['image_size'],
                                resize=self.resize)

        return {'image': [img_data],  'img_path': self.path_list[index]}

    def __len__(self):
        return len(self.data_list)




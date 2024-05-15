import pandas as pd
import numpy as np
import os,cv2,random
from torch.utils import data as data
from .tranforms import ratio_resize,paired_augment,single_augment
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from torchvision.transforms.functional import to_tensor


class ImgCataDataset(data.Dataset):
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
      super(ImgCataDataset, self).__init__()
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

      pd_file=pd.read_csv(opt['csv_path'])
      # for maximizing the io speed, the pre-process is applied in the initializing step
      # data_folder = opt['image_folder']
      # dir_list=os.listdir(self.data_folder)
      # for dir_name in dir_list:
      lq_l=pd_file['cataract-left'].dropna().to_list()
      lq_r=pd_file['cataract-right'].dropna().to_list()
      hq_l_raw=pd_file['normal-left'].dropna().to_list()
      hq_r_raw=pd_file['normal-right'].dropna().to_list()
      ratio=len(lq_l)//len(hq_l_raw)
      hq_imgs=[]
      for indx in range(ratio):
          hq_imgs.extend(hq_l_raw)
      hq_imgs.extend(random.sample(hq_l_raw,len(lq_l)%len(hq_l_raw)))
      ratio=len(lq_r)//len(hq_r_raw)
      # hq_imgs=[]
      for indx in range(ratio):
          hq_imgs.extend(hq_r_raw)
      hq_imgs.extend(random.sample(hq_r_raw,len(lq_r)%len(hq_r_raw)))
      lq_imgs=[]
      lq_imgs.extend(lq_l)
      lq_imgs.extend(lq_r)

      self.path_list=lq_imgs
      self.data_list=[]

      for indx in range(len(lq_imgs)):
          # print(lq_imgs[indx])
          # print(hq_imgs[indx])
          lq_im=Image.open(lq_imgs[indx])
          hq_im=Image.open(hq_imgs[indx])
          # sub_list=[]
          # sub_list.append(hq_im)
          # w,h=lq_im.size
          # if w<crop_size or h<crop_size:
          #     pass
          w_left=(hq_im.size[0]-lq_im.size[0])//2
          w_right=(hq_im.size[0]+lq_im.size[0])//2
          h_left=(hq_im.size[1]-lq_im.size[1])//2
          h_right=(hq_im.size[1]+lq_im.size[1])//2

          hq_crop=hq_im.crop((w_left, h_left, w_right, h_right))

          if self.min_multiplier!=1:
              # print('ratio resize activated!!')
              # print(self.resize)
              self.data_list.append([ratio_resize(lq_im,self.min_multiplier,opt['fine_size']),ratio_resize(hq_crop,self.min_multiplier,opt['fine_size']),ratio_resize(hq_im,self.min_multiplier,opt['fine_size'])])
          else:
              self.data_list.append([lq_im,hq_crop,hq_im])
          

    def __getitem__(self, index):

        hqc_data,lq_data=paired_augment(self.data_list[index][1].convert('RGB'),
                                       self.data_list[index][0].convert('RGB'),
                                       flip=self.opt['flip'], 
                                       fine_size=self.opt['fine_size'],
                                       crop=self.crop,
                                       crop_size=self.opt['image_size'],
                                       resize=self.resize)
        
        hq_data=single_augment(self.data_list[index][2].convert('RGB'),
                                flip=self.opt['flip'], 
                                fine_size=self.opt['fine_size'],
                                crop=self.crop,
                                crop_size=self.opt['image_size'],
                                resize=self.resize)

        return {'image': [lq_data,hqc_data,hq_data],  'img_path': self.path_list[index]}

    def __len__(self):
        return len(self.data_list)




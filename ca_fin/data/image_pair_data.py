import pandas as pd
import numpy as np
import os,cv2,random
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .tranforms import paired_augment,paired_random_crop,tensor2img
from PIL import Image


class ImagePairDataset(data.Dataset):
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
        super(ImagePairDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        self.crop = opt['crop'] if 'crop' in opt else False

        if self.mean is not None or self.std is not None:
          print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])

        pd_data = pd.DataFrame(raw_data)
        self.image_hq = pd_data['file_name'].tolist()
        self.image_lq = pd_data['file_name'].tolist()
        levels_raw = [pd_data['fr'].tolist(),pd_data['nc'].tolist(),pd_data['cc'].tolist(),pd_data['psc'].tolist()]
        self.levels=np.array(levels_raw).T
        

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_hq_list=[]
          new_lq_list=[]
          new_level_list=self.levels
          for times in range(0,self.augment_ratio-1):
            new_hq_list.extend(self.image_hq)
            new_lq_list.extend(self.image_lq)
            # print(self.levels.shape)
            # print(new_level_list.shape)
            new_level_list=np.concatenate([new_level_list,self.levels],axis=0)
          new_hq_list.extend(self.image_hq)
          new_lq_list.extend(self.image_lq)
          self.image_hq = new_hq_list
          self.image_lq = new_lq_list
          self.levels=new_level_list
        # print(self.levels.shape)
        # print(len(self.image_names))
        
        # # make the levels int
        # self.lvs= np.array(self.lvs).astype(int)



    def __getitem__(self, index):

        hq_name,suf=os.path.splitext(self.image_hq[index])
        hq_path = os.path.join(self.dt_folder, hq_name+"-hq.PNG")
        lq_path = os.path.join(self.dt_folder, self.image_lq[index])
        hq_data = Image.open(hq_path)
        lq_data = Image.open(lq_path)
        level = self.levels[index]
        

        # augment
        # auto reshape and crop into size
        hq_data,lq_data=paired_augment(hq_data.convert('RGB'),
                                       lq_data.convert('RGB'),
                                       flip=self.opt['flip'], 
                                       resize=self.opt['resize'],
                                       fine_size=self.opt['fine_size'],
                                       crop=self.crop,
                                       crop_size=self.opt['image_size'],
                                       )
        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(hq_data, self.mean, self.std, inplace=True)
            normalize(lq_data, self.mean, self.std, inplace=True)

        # saving=cv2.imwrite(os.path.join('/data/huden/CATINTELL/test_image',str(random.randint(0,20)))+'.PNG',tensor2img(hq_data))
        return {'hq': hq_data,'lq': lq_data, 'class': level,'lq_path': lq_path}

    def __len__(self):
        return len(self.image_hq)



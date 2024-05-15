import pandas as pd
import torch, time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from .base_model import BaseModel
from .build_utils import build_network,tensor2img,img_write
from .logger_utils import get_root_logger

class CatTestModel(BaseModel):
    """Catintell model for dehaze."""

    def __init__(self, opt):
        super(CatTestModel, self).__init__(opt)

        # define degrade network
        self.net_g_a = build_network(opt['network_g_A'])
        self.net_g_a = self.model_to_device(self.net_g_a)
        self.print_network(self.net_g_a)


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_net_g_A', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g_A', 'params')
            self.load_network(self.net_g_a, load_path, self.opt['path'].get('strict_load_g', True), param_key)


        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g_a.train()


        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_a_ema = build_network(self.opt['network_g_A']).to(self.device)

            load_path = self.opt['path'].get('pretrain_net_g_A', None)
            if load_path is not None:
                self.load_network(self.net_g_a_ema, load_path, self.opt['path'].get('strict_load_g_A', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_a_ema.eval()

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # set optimizer for model A
        if train_opt.get('optimizer_opt_A'):
          optimizer_opt=train_opt['optimizer_opt_A']

          optim_params = []
          for k, v in self.net_g_a.named_parameters():  # key ,value
              if v.requires_grad:
                  optim_params.append(v)
              else:
                  logger = get_root_logger()
                  logger.warning(f'Params {k} will not be optimized.')

          optim_type = optimizer_opt['optim_g'].pop('type')
          self.optimizer_g_a = self.get_optimizer(optim_type, optim_params, **optimizer_opt['optim_g'])
          self.optimizers.append(self.optimizer_g_a)

          # optimizer d
          optim_type = optimizer_opt['optim_d'].pop('type')
          self.optimizer_d_a = self.get_optimizer(optim_type, self.net_d_a.parameters(), **optimizer_opt['optim_d'])
          self.optimizers.append(self.optimizer_d_a)

          # set up iter number of g & d
          self.net_d_iters_a = optimizer_opt.get('net_d_iters', 1)
          self.net_g_init_iters_a = optimizer_opt.get('net_g_init_iters', 0)
          self.net_g_end_iters_a= optimizer_opt.get('net_g_end_iters', 99999999)

        

    def feed_data(self, data):
        self.cataract_image = data['image'][0].to(self.device)
        

    def optimize_parameters(self, current_iter):
        pass
    
    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g_a)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_a_ema.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)


    def test(self):
        if hasattr(self, 'net_g_a_ema'):
            self.net_g_a_ema.eval()
            with torch.no_grad():
                # prediction
                # self.output_a =  self.net_g_a_ema(self.cataract_image)
                self.output_a =  0.8*self.net_g_a_ema(self.cataract_image)+0.2*self.cataract_image
                self.output_a[self.cataract_image==0]=0
        else:
            self.net_g_a.eval()
            
            with torch.no_grad():
                # prediction
                self.output_a = self.net_g_a(self.cataract_image)
                # self.output_a =  0.8*self.net_g_a(self.cataract_image)+0.2*self.cataract_image
                self.output_a[self.cataract_image==0]=0
            self.net_g_a.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        if self.opt['val'].get('use_pbar'):
            self.use_pbar = self.opt['val']['use_pbar']
        else:
            self.use_pbar = False
        
        if self.use_pbar: 
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['img_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            
            lq_img=tensor2img(visuals['lq'])
            sr_img=tensor2img(visuals['result_a'])
            

            # tentative for out of GPU memory
            del self.cataract_image
            del self.output_a
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}')
                
                img_write(lq_img, save_img_path+'-cataract.png')
                img_write(sr_img, save_img_path+'-restore.png')

            
            if self.use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if self.use_pbar:
            pbar.close()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.cataract_image.detach().cpu()
        out_dict['result_a'] = self.output_a.detach().cpu()
        
        return out_dict
      

import pandas as pd
import torch, time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from .base_model import BaseModel
from .build_utils import build_network,build_loss,calculate_metric,csv_write,tensor2img,img_write
from .logger_utils import get_root_logger

class CatDualModel(BaseModel):
    """Catintell model for dehaze."""

    def __init__(self, opt):
        super(CatDualModel, self).__init__(opt)

        # define degrade network
        self.net_g_a = build_network(opt['network_g_A'])
        self.net_g_a = self.model_to_device(self.net_g_a)
        self.print_network(self.net_g_a)
        # define dehaze model
        self.net_g_b = build_network(opt['network_g_B'])
        self.net_g_b = self.model_to_device(self.net_g_b)
        self.print_network(self.net_g_b)

        # define two discriminators 
        self.net_d_a = build_network(opt['network_d_A'])
        self.net_d_a = self.model_to_device(self.net_d_a)
        self.net_d_b = build_network(opt['network_d_B'])
        self.net_d_b = self.model_to_device(self.net_d_b)
        # self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_net_g_A', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g_A', 'params')
            self.load_network(self.net_g_a, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_net_d_A', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d_A', 'params')
            self.load_network(self.net_d_a, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        load_path = self.opt['path'].get('pretrain_net_g_B', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g_B', 'params')
            self.load_network(self.net_g_b, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_net_d_B', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d_B', 'params')
            self.load_network(self.net_d_b, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g_a.train()
        self.net_g_b.train()
        self.net_d_a.train()
        self.net_d_b.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_a_ema = build_network(self.opt['network_g_A']).to(self.device)
            self.net_g_b_ema = build_network(self.opt['network_g_B']).to(self.device)

            load_path = self.opt['path'].get('pretrain_net_g_A', None)
            if load_path is not None:
                self.load_network(self.net_g_a_ema, load_path, self.opt['path'].get('strict_load_g_A', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_a_ema.eval()

            load_path = self.opt['path'].get('pretrain_net_g_B', None)
            if load_path is not None:
                self.load_network(self.net_g_b_ema, load_path, self.opt['path'].get('strict_load_g_B', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_b_ema.eval()

        # define losses of net A (degrade)
        if train_opt.get('loss_opt_A'):
          loss_opt=train_opt['loss_opt_A']
          if loss_opt.get('pixel_opt'):
              self.cri_pix_a = build_loss(loss_opt['pixel_opt']).to(self.device)
          else:
              self.cri_pix_a = None

          if loss_opt.get('perceptual_opt'):
              self.cri_per_a = build_loss(loss_opt['perceptual_opt']).to(self.device)
          else:
              self.cri_per_a = None

          if loss_opt.get('gan_opt'):
              self.cri_gan_a = build_loss(loss_opt['gan_opt']).to(self.device)
          else:
              self.cri_gan_a = None
          
          if loss_opt.get('iden_opt'):
              self.cri_iden_a = build_loss(loss_opt['iden_opt']).to(self.device)
          else:
              self.cri_iden_a = None

          if self.cri_per_a is None and self.cri_gan_a is None and self.cri_pix_a is None:
              raise ValueError('No loss found. Please use losses in train setting for model A (degradation models)')
        # define losses of net A (degrade)
        if train_opt.get('loss_opt_B'):
          loss_opt=train_opt['loss_opt_B']
          if loss_opt.get('pixel_opt'):
              self.cri_pix_b = build_loss(loss_opt['pixel_opt']).to(self.device)
          else:
              self.cri_pix_b = None

          if loss_opt.get('perceptual_opt'):
              self.cri_per_b = build_loss(loss_opt['perceptual_opt']).to(self.device)
          else:
              self.cri_per_b = None

          if loss_opt.get('gan_opt'):
              self.cri_gan_b = build_loss(loss_opt['gan_opt']).to(self.device)
          else:
              self.cri_gan_b = None

          if loss_opt.get('iden_opt'):
              self.cri_iden_b = build_loss(loss_opt['iden_opt']).to(self.device)
          else:
              self.cri_iden_b = None

          if self.cri_per_b is None and self.cri_gan_b is None and self.cri_pix_b is None:
              raise ValueError('No loss found. Please use losses in train setting for model B (restoration models)')


          # set up optimizers and schedulers
          # self.net_d_iters = train_opt.get('net_d_iters', 1)
          # self.net_g_init_iters = train_opt.get('net_g_init_iters', 0)
        
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

        # set optimizer for model B
        if train_opt.get('optimizer_opt_B'):
          optimizer_opt=train_opt['optimizer_opt_B']

          optim_params = []
          for k, v in self.net_g_b.named_parameters():  # key ,value
              if v.requires_grad:
                  optim_params.append(v)
              else:
                  logger = get_root_logger()
                  logger.warning(f'Params {k} will not be optimized.')

          optim_type = optimizer_opt['optim_g'].pop('type')
          self.optimizer_g_b = self.get_optimizer(optim_type, optim_params, **optimizer_opt['optim_g'])
          self.optimizers.append(self.optimizer_g_b)


          # optimizer d
          optim_type = optimizer_opt['optim_d'].pop('type')
          self.optimizer_d_b = self.get_optimizer(optim_type, self.net_d_b.parameters(), **optimizer_opt['optim_d'])
          self.optimizers.append(self.optimizer_d_b)


          # set up iter number of g & d
          self.net_d_iters_b = optimizer_opt.get('net_d_iters', 1)
          self.net_g_init_iters_b = optimizer_opt.get('net_g_init_iters', 0)
          self.net_g_end_iters_b= optimizer_opt.get('net_g_end_iters', 99999999)

    def feed_data(self, data):
        self.cataract_image = data['image'][0].to(self.device)
        self.normal_image = data['image'][1].to(self.device)
        self.normal_image_full = data['image'][2].to(self.device)
        
        # self.DR_grade = data['DR_grade'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g_a.zero_grad()

        # prediction
        self.output_a = self.net_g_a(self.normal_image)

        l_g_total = 0
        flag_a=False
        flag_b=False
        loss_dict = OrderedDict()
        if  current_iter < self.net_g_end_iters_a:
          if current_iter > self.net_g_init_iters_a:
            for p in self.net_d_a.parameters():
                p.requires_grad = False
            # pixel loss
            if self.cri_pix_a:
                l_g_pix = self.cri_pix_a(self.output_a, self.cataract_image)
                l_g_total += l_g_pix
                loss_dict['l_g_pix_A'] = l_g_pix
            # perceptual loss
            if self.cri_per_a:
                l_g_percep, l_g_style = self.cri_per_a(self.output_a, self.cataract_image)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep_A'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style_A'] = l_g_style
            if self.cri_iden_a:
                l_g_iden=self.cri_pix_a(self.net_g_a(self.cataract_image).detach(), self.cataract_image)
                l_g_total+=l_g_iden
                loss_dict['l_g_iden_A']=l_g_iden
            # gan loss
            fake_g_pred = self.net_d_a(self.output_a)
            l_g_gan = self.cri_gan_a(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_A'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g_a.step()


          # optimize net_d_a when end iteration is not achieved
          if current_iter % self.net_d_iters_a == 0:
            for p in self.net_d_a.parameters():
                p.requires_grad = True

            self.optimizer_d_a.zero_grad()
            # real
            real_d_pred = self.net_d_a(self.cataract_image)
            l_d_real = self.cri_gan_a(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real_a'] = l_d_real
            loss_dict['out_d_real_a'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d_a(self.output_a.detach())
            l_d_fake = self.cri_gan_a(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake_a'] = l_d_fake
            loss_dict['out_d_fake_a'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            
            self.optimizer_d_a.step()
        else:
            if not flag_a:
              for p in self.net_d_a.parameters():
                  p.requires_grad = False
              for p in self.net_g_a.parameters():
                  p.requires_grad = False
              flag_a=True

        

        ## prediction of dehaze model, it need the output of A
        self.optimizer_g_b.zero_grad()
        
        
        self.output_a=0.8*self.net_g_a(self.normal_image_full)+0.2*self.normal_image_full
        self.output_a[self.normal_image_full==0]=0
        self.output_b = self.net_g_b(self.output_a.detach())

        l_g_total = 0
        # loss_dict = OrderedDict()
        if current_iter < self.net_g_end_iters_b:
          if current_iter > self.net_g_init_iters_b:
            for p in self.net_d_b.parameters():
                p.requires_grad = False
            # pixel loss
            if self.cri_pix_b:
                l_g_pix = self.cri_pix_b(self.output_b, self.normal_image_full)
                l_g_total += l_g_pix
                loss_dict['l_g_pix_B'] = l_g_pix
            # perceptual loss
            if self.cri_per_b:
                l_g_percep, l_g_style = self.cri_per_b(self.output_b, self.normal_image_full)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep_B'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style_B'] = l_g_style
            if self.cri_iden_b:
                l_g_iden=self.cri_pix_b(self.net_g_b(self.normal_image_full).detach(), self.normal_image_full)
                l_g_total+=l_g_iden
                loss_dict['l_g_iden_B']=l_g_iden
            # gan loss
            fake_g_pred = self.net_d_b(self.output_b)
            l_g_gan = self.cri_gan_b(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_B'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g_b.step()

          # optimize net_d_a
          if current_iter % self.net_d_iters_b == 0:
            for p in self.net_d_b.parameters():
                p.requires_grad = True

            self.optimizer_d_b.zero_grad()
            # real
            real_d_pred = self.net_d_b(self.normal_image_full)
            l_d_real = self.cri_gan_b(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real_b'] = l_d_real
            loss_dict['out_d_real_b'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d_b(self.output_b.detach())
            l_d_fake = self.cri_gan_b(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake_b'] = l_d_fake
            loss_dict['out_d_fake_b'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            
            self.optimizer_d_b.step()
        else:
            if not flag_b:
              for p in self.net_d_b.parameters():
                  p.requires_grad = False
              for p in self.net_g_b.parameters():
                  p.requires_grad = False
              flag_b=True

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g_a)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_a_ema.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

        net_g = self.get_bare_model(self.net_g_b)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_b_ema.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def test(self):
        if hasattr(self, 'net_g_a_ema'):
            self.net_g_a_ema.eval()
            self.net_g_b_ema.eval()
            with torch.no_grad():
                # prediction
                self.output_a =  0.8*self.net_g_a_ema(self.normal_image_full)+0.2*self.normal_image_full
                self.output_a[self.normal_image_full==0]=0
                self.output_b =  self.net_g_b_ema(self.output_a.detach())

        else:
            self.net_g_a.eval()
            self.net_g_b.eval()
            with torch.no_grad():
                # prediction
                self.output_a = 0.8*self.net_g_a(self.normal_image_full)+0.2*self.normal_image_full
                self.output_a[self.normal_image_full==0]=0
                self.output_b = self.net_g_b(self.output_a.detach())

            self.net_g_a.train()
            self.net_g_b.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        if self.opt['val'].get('use_pbar'):
            self.use_pbar = self.opt['val']['use_pbar']
        else:
            self.use_pbar = False

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if self.use_pbar: 
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['img_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            
            hq_img = tensor2img(visuals['hq'])
            lq_img=tensor2img(visuals['lq'])
            de_img=tensor2img(visuals['result_a'])
            sr_img = tensor2img(visuals['result_b'])
            

            # tentative for out of GPU memory
            del self.normal_image
            del self.cataract_image
            del self.output_a
            del self.output_b
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
                
                img_write(hq_img, save_img_path+'-normal.png')
                img_write(lq_img, save_img_path+'-cataract.png')
                img_write(de_img, save_img_path+'-degrade.png')
                img_write(sr_img, save_img_path+'-restore.png')

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=hq_img)
                    # print(name,opt_,metric_data)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if self.use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if self.use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['hq'] = self.normal_image_full.detach().cpu()
        out_dict['lq'] = self.cataract_image.detach().cpu()
        # print(out_dict['lq'].shape)
        out_dict['result_a'] = self.output_a.detach().cpu()
        out_dict['result_b'] = self.output_b.detach().cpu()
        # print(out_dict['result'].shape)
        
        return out_dict
        
        
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        # print(self.metric_results.items())
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter,multi_round=0):
        if hasattr(self, 'net_g_ema_a'):
            self.save_network([self.net_g_a, self.net_g_a_ema], 'net_g_a', current_iter,round=multi_round, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g_a, 'net_g_a', current_iter,round=multi_round)
        if hasattr(self, 'net_g_ema_b'):
            self.save_network([self.net_g_b, self.net_g_b_ema], 'net_g_b', current_iter,round=multi_round, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g_b, 'net_g_b', current_iter,round=multi_round)

        # self.save_network(self.net_d_a, 'net_d_a', current_iter,round=multi_round)
        # self.save_network(self.net_d_b, 'net_d_b', current_iter,round=multi_round)
        self.save_training_state(epoch, current_iter)


    # rewrite the saving
    def save_network(self, net, net_label, current_iter,round, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{round}_{net_label}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

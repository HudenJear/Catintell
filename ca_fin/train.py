import datetime,random,copy
import logging
import math
import time,os,glob
# from matplotlib.pyplot import axis
import torch
from os import path as osp
import pandas as pd
import numpy as np
# import ssl
# import urllib.request
# ssl._create_defalut_https_context=ssl._create_unverified_context


# from data import build_dataloader, build_dataset
# from data.data_sampler import EnlargedSampler
from data.fetcher import CPUPrefetcher
from data.data_utils import build_dataloader,build_dataset
from data.sampler import EnlargedSampler
from model.build_utils import check_resume,make_exp_dirs,scandir
from model.logger_utils import get_env_info,get_root_logger,AvgTimer, MessageLogger,dict2str
from model.option_utils import parse_options, copy_opt_file
from model.build_model import build_model

# # import archs
# from utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
#                            init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
# from utils.options import copy_opt_file, dict2str, parse_options



def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            logger.info(f'Dataset [{train_set.__class__.__name__}] - {dataset_opt["name"]} ' 'is built.')
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

def fetch_train_val_dataloader(round_num, opt, logger):
    # random fetch a new train and val dataloaders
    #csv will be saved in models folder
    train_loader, val_loaders = None, []
    dir_list= os.listdir(opt['random_fetch_path'])
    train_dict={}
    val_dict={}
    for dir_name in dir_list:
       img_list=glob.glob(os.path.join(opt['random_fetch_path'],dir_name,'*.PNG'))
       random.shuffle(img_list) 
       train_dict[dir_name]=img_list[int(len(img_list)*opt['val_ratio']):]
       val_dict[dir_name]=img_list[0:int(len(img_list)*opt['val_ratio'])]
    
    # load data
    test_pd=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in val_dict.items()]))
    train_pd=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in train_dict.items()]))

    csv_save_path= osp.join('experiments', opt['name'], 'models')
    if not osp.exists(csv_save_path):
       os.mkdir(csv_save_path)
    saving1=train_pd.to_csv(osp.join(csv_save_path,f'{round_num}_train.csv'))
    saving2=test_pd.to_csv(osp.join(csv_save_path,f'{round_num}_test.csv'))

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            dataset_opt['csv_path']=osp.join(csv_save_path,f'{round_num}_train.csv')
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            dataset_opt['csv_path']=osp.join(csv_save_path,f'{round_num}_test.csv')
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters



def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path


  # multi round train
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")
    logger = get_root_logger(logger_name='Catintell', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # set the multi round
    if opt['multitrain']:
      train_round=opt['round']
      logger.info(f'Model will be trained for {train_round} times')
    else:
      train_round=1
      logger.info(f'Model will be trained for {train_round} times')

    store_best=[]
    opt_back=opt
    for multi_round in range(0,train_round):
      opt=copy.deepcopy(opt_back)
      # create train and validation dataloaders
      if opt['dataset_random_fetch']:
        result=fetch_train_val_dataloader(multi_round, opt, logger)
      else:
        result = create_train_val_dataloader(opt, logger)
      train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

      # create model
      model = build_model(opt)
      if resume_state:  # resume training
          model.resume_training(resume_state)  # handle optimizers and schedulers
          logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
          start_epoch = resume_state['epoch']
          current_iter = resume_state['iter']
      else:
          start_epoch = 0
          current_iter = 0

      # create message logger (formatted outputs)
      msg_logger = MessageLogger(opt, current_iter)

      # dataloader prefetcher
      # prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
      
      prefetcher = CPUPrefetcher(train_loader)
      
      # short examination of the parameter and flops using thop, require the thop package
      # Not necessary for the normal training
      # cited in normal training !!!

      # from thop import profile
      # input = torch.randn(1, 3, 384, 384).cuda()
      # flops, params = profile(model.net_g, inputs=(input, ))
      # logger.info(f'Detecting the model parameter and computing requirement...\nFlops: {flops}\nParas:{params}')


      # training
      logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
      data_timer, iter_timer = AvgTimer(), AvgTimer()
      start_time = time.time()

      # the saving model mode was changed into save current best only
      # this is very important for long term training
      # saving quite some storage!!!
      save_best_mode=True if opt['logger']['save_best'] else False
      last_best={}



      # begin
      for epoch in range(start_epoch, total_epochs + 1):
          train_sampler.set_epoch(epoch)
          prefetcher.reset()
          train_data = prefetcher.next()

          while train_data is not None:
              data_timer.record()

              current_iter += 1
              if current_iter > total_iters:
                  break
              # update learning rate
              model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
              # training
              model.feed_data(train_data)
              model.optimize_parameters(current_iter)
              iter_timer.record()
              if current_iter == 1:
                  # reset start time in msg_logger for more accurate eta_time
                  # not work in resume mode
                  msg_logger.reset_start_time()
              # log
              if current_iter % opt['logger']['print_freq'] == 0:
                  log_vars = {'epoch': epoch, 'iter': current_iter}
                  log_vars.update({'lrs': model.get_current_learning_rate()})
                  log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                  log_vars.update(model.get_current_log())
                  msg_logger(log_vars)

              # validation
              if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                  if len(val_loaders) > 1:
                      logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                  for val_loader in val_loaders:
                      model.validation(val_loader, current_iter, opt['val']['save_img'], tb_logger=None)

              # save models and training states
              # the saving model mode was changed into save current best only
              # this is very important for long term training
              # saving quite some storage!!!
              if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                if save_best_mode:
                  if last_best==model.best_metric_results:
                    logger.info('Nothing better. Not saving models and training states.')
                  else:
                    logger.info('Found better metrics! Saving models and training states.')
                    model.save(epoch, current_iter, multi_round)
                    last_best=copy.deepcopy(model.best_metric_results)

                else:
                  logger.info('Saving models and training states.')
                  model.save(epoch, current_iter, multi_round)

              data_timer.start()
              iter_timer.start()
              train_data = prefetcher.next()
          # end of iter
      # end of epoch

      consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
      logger.info(f'End of training. Time consumed: {consumed_time}')
      logger.info('Save the latest model.')
      model.save(epoch=-1, current_iter=-1,multi_round=multi_round)  # -1 stands for the latest
      if opt.get('val') is not None:
          for val_loader in val_loaders:
              model.validation(val_loader, current_iter, opt['val']['save_img'], tb_logger=None)
      store_best.append(model.best_metric_results)
    get_best_median(store_best,logger)


def get_best_median(best_collect,logger):
  sub_best=best_collect[0]
  # build a list for all of the metric results
  for dataset, dataset_value in sub_best.items():
    # print(dataset)
    metrics=[]
    better=[]
    mat_value=[]
    iter_value=[]
    for metric,met_values in dataset_value.items():
      metrics.append(metric)
      better.append(sub_best[dataset][metric]["better"])
      mat_value.append([])
      iter_value.append([])
      # logger.info(f'{metric}'
      #                   f'\tBest: {best[dataset][metric]["val"]:.4f} @ '
      #                   f'{best[dataset][metric]["iter"]} iter')
  # collect the values
  for best in best_collect:
      for dataset, dataset_value in best.items():
        # print(dataset)
        for ind in range(0,len(metrics)):
          mat_value[ind].append(best[dataset][metrics[ind]]["val"])
          iter_value[ind].append(best[dataset][metrics[ind]]["iter"])

        for metric,met_values in dataset_value.items():
          logger.info(f'\tBest: {best[dataset][metric]["val"]:.4f} @ '
                            f'{best[dataset][metric]["iter"]} iter'
                            f'{metric}')
      # logger.info('\n')
  results=np.array(mat_value)
  medians=np.median(results,axis=1)
  log_str1='Train val results:\n'
  log_str2='Median results (even times trainning would use an average of median 2 results):\n'
  for ind in range(0,len(metrics)):
    if better[ind]=='higher':
      best_ind=np.argmax(results[ind])
    else:
      best_ind=np.argmin(results[ind])
    log_str1+='Best '+metrics[ind]+': {} @ iter {} in train {}\n'.format(mat_value[ind][best_ind],iter_value[ind][best_ind],best_ind)
    log_str2+='Median '+metrics[ind]+': {}\n'.format(medians[ind])

  logger.info(log_str1)
  logger.info(log_str2)

  return





if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)

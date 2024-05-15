from copy import deepcopy
import os ,time ,cv2,torch,math
from torchvision.utils import make_grid
import numpy as np
from .arch.swin_transformer_arch import SwinTransformer
from .arch.ConvNeXt_arch import ConvNeXt
from .arch.Catintell_D5_arch import CatintellConv_D5,Catintell_G5
from .arch.RRDB_arch import RRDBNet
from .logger_utils import get_root_logger
from .loss.losses import PerceptualLoss,GANLoss,GANFeatLoss,L1Loss,MSELoss,SmoothL1Loss,FundusClassLoss,FundusPerceptualLoss
from .metrics.class_metrics import calculate_acc,calculate_f1,calculate_p,calculate_r
from .metrics.image_metrics import calculate_psnr,calculate_ssim,calculate_l1

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    logger = get_root_logger()
    if network_type in ['CatintellConv_D5','ConvNeXt','SwinTransformer','RRDBNet','Catintell_G5']:

      if network_type=='SwinTransformer':
        net = SwinTransformer(**opt)
      elif network_type=='ConvNeXt':
        net = ConvNeXt(**opt)
      elif network_type=='RRDBNet':
        net = RRDBNet(**opt)
      elif network_type=='CatintellConv_D5':
        net = CatintellConv_D5(**opt)
      elif network_type=='Catintell_G5':
        net = Catintell_G5(**opt)
      logger.info(f'Network [{net.__class__.__name__}] is created.')
    else:
      net = None
      logger.info(f'Network is NOT created. No matched name.')
    
    return net

def build_loss(opt):
  opt = deepcopy(opt)
  loss_type = opt.pop('type')
  logger = get_root_logger()
  if loss_type in ['PerceptualLoss','GANLoss','GANFeatLoss','L1Loss','SmoothL1Loss','MSELoss','FundusClassLoss','FundusPerceptualLoss']:
    if loss_type=='PerceptualLoss':
      new_loss = PerceptualLoss(**opt)
    elif loss_type=='GANLoss':
      new_loss = GANLoss(**opt)
    elif loss_type=='GANFeatLoss':
      new_loss =GANFeatLoss(**opt)
    elif loss_type=='L1Loss':
      new_loss =L1Loss(**opt)
    elif loss_type=='SmoothL1Loss':
      new_loss =SmoothL1Loss(**opt)
    elif loss_type=='MSELoss':
      new_loss =MSELoss(**opt)
    elif loss_type=='FundusClassLoss':
      new_loss =FundusClassLoss(**opt)
    elif loss_type=='FundusPerceptualLoss':
      new_loss =FundusPerceptualLoss(**opt)
    logger.info(f'Loss [{new_loss.__class__.__name__}] is created.')
  else:
    new_loss = None
    logger.info(f'Loss Function '+loss_type+' is NOT created. No matched name.')

  return new_loss



def calculate_metric(data, opt):
  """Calculate metric from data and options.

  Args:
      opt (dict): Configuration. It must contain:
          type (str): Model type.
  """
  opt = deepcopy(opt)
  metric_type = opt.pop('type')
  if metric_type in ['calculate_acc','calculate_f1','calculate_p','calculate_r','calculate_psnr','calculate_ssim','calculate_l1']:
    if metric_type=='calculate_acc':
      result = calculate_acc(**data,**opt)
    elif metric_type=='calculate_f1':
      result = calculate_f1(**data,**opt)
    elif metric_type=='calculate_p':
      result =calculate_p(**data,**opt)
    elif metric_type=='calculate_r':
      result =calculate_r(**data,**opt)
    elif metric_type=='calculate_psnr':
      result =calculate_psnr(**data,**opt)
    elif metric_type=='calculate_ssim':
      result =calculate_ssim(**data,**opt)
    elif metric_type=='calculate_l1':
      result =calculate_l1(**data,**opt)
  return result

def csv_write(data_frame, file_path, params=None, auto_mkdir=True):
  """Write csv to file.

  Args:
      data_frame (pd.DataFrame): csv data.
      file_path (str): saving file path.
      params (None or list): Same as to_csv() interference.
      auto_mkdir (bool): If the parent folder of `file_path` does not exist,
          whether to create it automatically.
  """
  if auto_mkdir:
      dir_name = os.path.abspath(os.path.dirname(file_path))
      os.makedirs(dir_name, exist_ok=True)
  sav = data_frame.to_csv(file_path)

  
def img_write(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')

def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            print('pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (network
                                                                     not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = os.path.join(opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                print(f"Set {name} to {opt['path'][name]}")

        # change param_key to params in resume
        param_keys = [key for key in opt['path'].keys() if key.startswith('param_key')]
        for param_key in param_keys:
            if opt['path'][param_key] == 'params_ema':
                opt['path'][param_key] = 'params'
                print(f'Set {param_key} to params')


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


# @master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' in key) or ('pretrain_net' in key) or ('resume' in key) or ('param_key' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)



def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



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



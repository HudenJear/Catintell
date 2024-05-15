# from .catintell_dehaze_model import DehazeModel
# from .catintell_generate_model import GenerateModel
from .catintell_model import CatDualModel
from .catintell_inference_model import CatTestModel
from .logger_utils import get_root_logger
from copy import deepcopy

def build_model(opt):
  """Build model from options.

  Args:
      opt (dict): Configuration. It must contain:
          model_type (str): Model type.
  """
  model_type = opt.pop('model_type')
  opt = deepcopy(opt)
  if model_type in ['CatDualModel','CatTestModel']:
    if model_type == 'CatDualModel':
      model = CatDualModel(opt)
    elif model_type == 'CatTestModel':
      model = CatTestModel(opt)
    # elif model_type == 'SequentialModel':
    #   model = SequentialModel(opt)
    # elif model_type == 'ComboModel':
    #   model = ComboModel(opt)
    # elif model_type == 'HyperIQAModel':
    #   model = HyperIQAModel(opt)
    # elif model_type == 'CASRModel':
    #   model = CASRModel(opt)
    # elif model_type == 'Class2Model':
    #   model = Class2Model(opt)
    # elif model_type == 'MultiClassModel':
    #   model = MultiClassModel(opt)
    
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
  else: 
    model =None
    logger = get_root_logger()
    logger.info('Model '+model_type+' is NOT created. No matched name.')
  
  return model
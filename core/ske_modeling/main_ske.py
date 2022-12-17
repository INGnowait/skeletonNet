# coding : utf-8
import logging
import os
import mmcv
import os.path as osp
import numpy as np
import sys
from setproctitle import setproctitle
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.engine import default_setup, launch
from mmcv import Config
import cv2
import pickle
import scipy.io as scio

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# try:
#     import horovod.torch as hvd
# except ImportError:
#     print("You requested to import horovod which is missing or not supported for your OS.")

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from core.utils import my_comm as comm

from lib.utils.utils import iprint
from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str

from core.ske_modeling.dataset_factory import register_datasets_in_cfg
from core.ske_modeling.engine import do_test, do_train
from core.ske_modeling.models import SKEN


logger = logging.getLogger("detectron2")


def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename

def setup(args):
    """Create configs and perform basic setups."""
    method = 'sken'
    if 'sken' != method:
        cfg = Config.fromfile(args.config_file)
        if args.opts is not None:
            cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # NOTE: check if need to set OUTPUT_DIR automatically
        if cfg.OUTPUT_DIR.lower() == "auto":
            cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_ROOT, osp.splitext(args.config_file)[0].split("configs/")[1])
            iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

        if cfg.get("EXP_NAME", "") == "":
            setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
        else:
            setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

        if cfg.SOLVER.AMP.ENABLED:
            if torch.cuda.get_device_capability() <= (6, 1):
                iprint("Disable AMP for older GPUs")
                cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanterd configs in detectron2
        cfg.SOLVER.pop("STEPS", None)
        cfg.SOLVER.pop("MAX_ITER", None)
    # NOTE: get optimizer from string cfg dict
        if cfg.SOLVER.OPTIMIZER_CFG != "":
            if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
                optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            else:
                optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
            iprint("optimizer_cfg:", optim_cfg)
            cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
            cfg.SOLVER.BASE_LR = optim_cfg["lr"]
            cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
            cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        if cfg.get("DEBUG", False):
            iprint("DEBUG")
            args.num_gpus = 1
            args.num_machines = 1
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.TRAIN.PRINT_FREQ = 1

        exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

        if args.eval_only:
            if cfg.TEST.USE_PNP:
                # NOTE: need to keep _test at last
                exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
            else:
                exp_id += "_test"
        cfg.EXP_ID = exp_id
        cfg.RESUME = args.resume
        ####################################
        """save cfg and print some system information"""
        my_default_setup(cfg, args) # save cfg and print some system & settings information
    else:
        cfg = Config.fromfile( osp.normpath(osp.join(cur_dir, "../.."))+'/configs/sken/air_flight/sken_base.py')
    # register datasets
        """ 注册的是lm_dataset_d2.py中SPILITS_LM.lm_13_train中的除.ref_key的信息，并将lm中的15个目标变为13个，添加了dict，对应id和目标名称。
        同时还增加了evaluation的一些设置，具体请看lm_dataset_d2.py中函数register_with_name_cfg关于evaluation的注释
    """
    register_datasets_in_cfg(cfg)


    ####################################
    # Setup logger
    setup_for_distributed(is_master=comm.is_main_process())
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="core")
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lib")
    return cfg

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def main(args):
    cfg = setup(args)

    logger.info(f"Used CDPN module name: {cfg.MODEL.CDPN.NAME}")
    model, optimizer = eval(cfg.MODEL.CDPN.NAME).build_model_optimizer(cfg) # 没有rot_xyz和rot_region，用skeleton和keypoints替换
    logger.info("Model:\n{}".format(model))
    print('params:%.3fM' % count_parameters_in_MB(model))

    if args.eval_only:
        if cfg.MODEL.WEIGHTS == '':
            cfg.MODEL.WEIGHTS = '/media/j/data/skeletonNet/core/ske_modeling/output/sken/air_flight/sken_base/model_final.pth'
        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed and not args.use_hvd:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, args, model, optimizer, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    import resource

    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(500000, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    args = my_default_argument_parser().parse_args()
    iprint("Command Line Args:", args)

    if args.eval_only:
        torch.multiprocessing.set_sharing_strategy("file_system")

    USE_HVD = False
    if args.use_hvd:
        if comm.HVD_AVAILABLE:
            iprint("Using horovod")
            comm.init_hvd()
            USE_HVD = True
            main(args)
        else:
            iprint("horovod is not available. Fall back to default setting.")

    if not USE_HVD:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

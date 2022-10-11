# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('--config', default='projects/configs/nvformer/detr3d_res50_gridmask_cbgs_4subset.py', help='test config file path')
    parser.add_argument('--result', default='work_dirs/results.pkl', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', default='internal_code/figs', help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    import sys
    sys.path.append('./')
    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)
    eval_pipeline = cfg.get('eval_pipeline', {})
    if eval_pipeline:
        dataset.show_panorama(results, args.show_dir, pipeline=eval_pipeline)
    else:
        dataset.show_panorama(results, args.show_dir)  # use default pipeline


if __name__ == '__main__':
    main()

""" run_mot_challenge.py
    # for two scenarios comined: mot and mot with segmentation. More detail in 'SPECIFY_DATASET' option.

Run example:
run_mot_challenge.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL MPNTrack

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'SPECIFY_DATASET': None # Required: 'MOT' or 'MOTS'. Choose the working dataset for
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
        'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/MOTS-split_to_eval)
        'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
        'SKIP_SPLIT_FOL': False,    # If False, data is in GT_FOLDER/MOTS-SPLIT_TO_EVAL/ and in
                                    # TRACKERS_FOLDER/MOTS-SPLIT_TO_EVAL/tracker/
                                    # If True, then the middle 'MOTS-split' folder is skipped for both.
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE', 'JAndF']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

from trackeval.metrics import MaxSim
print(MaxSim)
if __name__ == '__main__':
    freeze_support()

    ######## My CODE HERE ########
    parser = argparse.ArgumentParser()
    parser.add_argument('--SPECIFY_BENCHMARK', required='True', choices=['KITTI', 'CHALLENGE'], help="Use one of the options to specify the applied data benchmark: 'KITTI' or 'CHALLENGE'.")
    parser.add_argument('--SPECIFY_DATASET', required='True', choices=['MOT', 'MOTS'], help="Please, use one of the options: 'MOT' or 'MOTS'.")
    args, unknown = parser.parse_known_args()
    benchmark, dataset_t = args.SPECIFY_BENCHMARK, args.SPECIFY_DATASET

    if benchmark == 'CHALLENGE':
        if dataset_t == "MOT":
            default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
        elif dataset_t == "MOTS":
            default_dataset_config = trackeval.datasets.MOTSChallenge.get_default_dataset_config()
            default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE', 'JAndF']}
        add_extra_config = {'SPECIFY_DATASET': dataset_t, 'SPECIFY_BENCHMARK': benchmark}
    else:

        if dataset_t == "MOT":
            default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
            GT_FOLDER = 'data/gt/kitti/kitti_2d_box_train'
        elif dataset_t == "MOTS":
            default_dataset_config = trackeval.datasets.KittiMOTS.get_default_dataset_config()
            GT_FOLDER = 'data/gt/kitti/kitti_mots_train'
        add_extra_config = {'SPECIFY_DATASET': dataset_t, 'GT_FOLDER': GT_FOLDER, 'SPECIFY_BENCHMARK': benchmark}
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}


    ########################################################

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config, **add_extra_config}  # Merge default configs
    for setting in config.keys():
        if setting not in ["SPECIFY_DATASET", "SPECIFY_BENCHMARK"]:
                if type(config[setting]) == list or type(config[setting]) == type(None):
                    parser.add_argument("--" + setting, nargs='+')
                else:
                    parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if benchmark == 'CHALLENGE':
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)] if dataset_t == "MOT" else [trackeval.datasets.MOTSChallenge(dataset_config)]
    else:
        dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)] if dataset_t == "MOT" else [trackeval.datasets.KittiMOTS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE, trackeval.metrics.JAndF, trackeval.metrics.MaxSim]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
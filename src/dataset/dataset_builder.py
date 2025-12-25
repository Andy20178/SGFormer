#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from xml.dom.minidom import Node
from src.dataset.dataset_3dssg import SSGDatasetGraph

def build_dataset(config, split_type, shuffle_objs, multi_rel_outputs,
                  use_rgb, use_normal, task_id=0, continue_learning_mode=None, is_all_task=False):
    if split_type != 'train_scans' and split_type != 'validation_scans' and split_type != 'test_scans':
        raise RuntimeError(split_type)
    if config.continue_learning_mode is not None:
        task_id = task_id
        continue_learning_mode = config.continue_learning_mode
    dataset = SSGDatasetGraph(
        config,
        split=split_type,
        multi_rel_outputs=multi_rel_outputs,
        shuffle_objs=shuffle_objs,
        use_rgb=use_rgb,
        use_normal=use_normal,
        label_type='3RScan160',
        for_train=split_type == 'train_scans',
        max_edges = config.dataset.max_edges,
        task_id=task_id,
        continue_learning_mode=continue_learning_mode,
        is_all_task=is_all_task
    )
    return dataset


if __name__ == '__main__':
    from config import Config
    config = Config('../config_example.json')
    config.dataset.root = '../data/example_data'
    build_dataset(config, split_type = 'train_scans', shuffle_objs=True, multi_rel_outputs=False,use_rgb=True,use_normal=True)
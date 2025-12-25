import numpy as np
import torch
import torch.nn.functional as F
import json

def get_gt(objs_target, rels_target, edges, multi_rel_outputs):
    gt_edges = []
    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0]
        idx_os = edges[edge_index][1]
        target_eo = objs_target[idx_eo]
        target_os = objs_target[idx_os]
        target_rel = []
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.shape[-1]):
                if rels_target[edge_index][i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0: # not None
                target_rel.append(rels_target[edge_index])
        gt_edges.append((target_eo, target_os, target_rel))
    return gt_edges


def evaluate_topk_object(objs_pred, objs_target, topk):
    res = []
    for obj in range(len(objs_pred)):
        obj_pred = objs_pred[obj]
        sorted_idx = torch.sort(obj_pred, descending=True)[1]
        gt = objs_target[obj]
        index = 1
        for idx in sorted_idx:
            if obj_pred[gt] >= obj_pred[idx] or index > topk:
                break
            index += 1
        res.append(index)
    return np.asarray(res)


def evaluate_topk_predicate(rels_preds, gt_edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res = []
    for rel in range(len(rels_preds)):
        rel_pred = rels_preds[rel]
        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_pred.max() < confidence_threshold:
        #         rel_pred[0] = rel_pred.max() + epsilon
        
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel][2]
        
        if len(rels_target) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            
            temp_topk.append(index)

        for gt in rels_target:
            index = 1
            for idx in sorted_idx:
                if rel_pred[gt] >= rel_pred[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
    # import pdb;pdb.set_trace()
    return np.asarray(res)
def evaluate_topk_predicate_with_per_class(rels_preds, gt_edges, multi_rel_outputs, topk, 
                                         confidence_threshold=0.5, epsilon=0.02):
    """
    评估top-k关系预测准确率，并返回每个类别的统计信息
    
    Returns:
        res: 原始的top-k排名结果
        per_class_stats: 每个类别的统计信息
    """
    res = []
    # 初始化每个类别的统计信息
    max_classes = len(rels_preds[0]) if len(rels_preds) > 0 else 0
    per_class_stats = {
        i: {
            'total_samples': 0,      # 总样本数
            'support': 0,            # 该类别作为正样本的次数
            'correct_top1': 0,       # top-1预测正确的次数
            'correct_topk': 0,       # top-k预测正确的次数（召回率计算）
            'none_as_positive': 0,   # 无关系时预测为该类的次数（误报）
        } for i in range(max_classes)
    }
    
    for rel_idx, rel_pred in enumerate(rels_preds):
        # 复制预测结果
        rel_pred_copy = rel_pred.clone()
        
        # 处理none类的特殊逻辑
        if multi_rel_outputs:
            if rel_pred_copy.max() < confidence_threshold:
                rel_pred_copy[0] = rel_pred_copy.max() + epsilon
        
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred_copy, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel_idx][2] if len(gt_edges[rel_idx]) > 2 else []
        
        # 更新每个类别的统计信息
        for class_id in range(max_classes):
            per_class_stats[class_id]['total_samples'] += 1
            
            # 统计support（该类别在真实标签中出现的次数）
            if class_id in rels_target:
                per_class_stats[class_id]['support'] += 1
            
            # 统计误报：无关系时预测为该类
            if len(rels_target) == 0 and sorted_idx[0].item() == class_id:
                per_class_stats[class_id]['none_as_positive'] += 1
        
        # Top-1准确率统计
        pred_top1 = sorted_idx[0].item()
        if len(rels_target) == 0:
            # 无真实关系时，预测为none类（0）是正确的
            if pred_top1 == 0:
                per_class_stats[0]['correct_top1'] += 1
        else:
            # 有真实关系时，预测的top-1类别在真实标签中则是正确的
            if pred_top1 in rels_target:
                per_class_stats[pred_top1]['correct_top1'] += 1
        
        # Top-k召回率统计
        topk_indices = sorted_idx[:topk].tolist()
        if len(rels_target) == 0:
            # 无真实关系时，如果none类（0）在top-k中则算正确预测
            if 0 in topk_indices:
                per_class_stats[0]['correct_topk'] += 1
        else:
            # 有真实关系时，每个真实关系类别在top-k中都算正确召回
            for gt_class in rels_target:
                if gt_class in topk_indices:
                    per_class_stats[gt_class]['correct_topk'] += 1
        
        # 原始排名计算逻辑（保持不变）
        if len(rels_target) == 0:
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item() + 1
            temp_topk.append(index)
        else:
            for gt in rels_target:
                index = 1
                for idx in sorted_idx:
                    if rel_pred_copy[gt] >= rel_pred_copy[idx] or index > topk:
                        break
                    index += 1
                temp_topk.append(index)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
    
    return np.asarray(res), per_class_stats

def evaluate_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res, cls = [], []
    # convert the score from log_softmax to softmax
    objs_pred = np.exp(objs_pred)
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        obj = objs_pred[edge_from]
        sub = objs_pred[edge_to]

        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_predictions.max() < confidence_threshold:
        #         rel_predictions[0] = rel_predictions.max() + epsilon

        size_o = len(obj)
        size_r = len(rel_predictions)

        node_score = np.matmul(obj.reshape(size_o, 1), sub.reshape(1, size_o))
        conf_matrix = np.matmul(node_score.reshape(size_o, size_o, 1), rel_predictions.reshape(1, size_r))
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)[1]

        subject = gt_rel[edge][0]
        obj = gt_rel[edge][1]
        temp_topk, tmp_cls = [], []

        for predicate in gt_rel[edge][2]:
            index = 1
            for idx_1d in sorted_args_1d:
                idx = np.unravel_index(idx_1d, (size_o, size_o, size_r))
                gt_conf = conf_matrix[subject, obj, predicate]
                if gt_conf >= conf_matrix[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
            tmp_cls.append(predicate)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            assert (tmp - counter) > 0
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
        cls += tmp_cls
    
    return np.asarray(res), np.array(cls)


def evaluate_triplet_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02, use_clip=False, obj_topk=None):
    res, triplet = [], []
    if not use_clip:
        objs_pred = np.exp(objs_pred)
    else:   
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    sub_scores, obj_scores, rel_scores = [],  [],  []
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]
        
        if obj_topk is not None:
            sub_pred = obj_topk[edge_from]
            obj_pred = obj_topk[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
        
        # just take topk
        sorted_conf_matrix = sorted_conf_matrix[:topk]
        sorted_args_1d = sorted_args_1d[:topk]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]
        temp_topk, tmp_triplet = [], []

        if len(rel_gt) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, -1])
            else:
                tmp_triplet.append([sub_gt.cpu(),obj_gt.cpu(),-1])
        
        for predicate in rel_gt: # for multi class case
            gt_conf = conf_matrix[sub_gt, obj_gt, predicate]
            indices = torch.where(sorted_conf_matrix == gt_conf)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, predicate])
            else:
                tmp_triplet.append([sub_gt.cpu(), obj_gt.cpu(), predicate])
            
            sub_scores.append(sub)
            obj_scores.append(obj)
            rel_scores.append(rel_predictions)
            
   
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        triplet += tmp_triplet
    # import pdb;pdb.set_trace()
    return np.asarray(res), np.array(triplet), sub_scores, obj_scores, rel_scores


def evaluate_topk_recall(objs_pred, rels_pred, objs_target, rels_target, edges):
    top_k_obj = evaluate_topk_object(objs_pred, objs_target, topk=10)
    gt_edges = get_gt(objs_target, rels_target, edges, topk=10)
    top_k_predicate = evaluate_topk_predicate(rels_pred, gt_edges, multi_rel_outputs=True, topk=5)
    top_k = evaluate_triplet_topk(objs_pred, rels_pred, rels_target, edges, multi_rel_outputs=True, topk=100)
    return top_k, top_k_obj, top_k_predicate


def get_mean_recall(triplet_rank, cls_matrix, topk=[50, 100]):
    if len(cls_matrix) == 0:
        return np.array([0,0])

    mean_recall = [[] for _ in range(len(topk))]
    cls_num = int(cls_matrix.max())
    for i in range(cls_num):
        cls_rank = triplet_rank[cls_matrix[:,-1] == i]
        if len(cls_rank) == 0:
            continue
        for idx, top in enumerate(topk):
            mean_recall[idx].append((cls_rank <= top).sum() * 100 / len(cls_rank))
    mean_recall = np.array(mean_recall, dtype=np.float32)
    return mean_recall.mean(axis=1)


def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output


def read_json(split, root):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list(f'{root}/train_scans.txt'))
        with open(f'{root}/relationships_train.json', "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list(f'{root}/validation_scans.txt'))
        with open(f'{root}/relationships_validation.json', "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    return data

def get_zero_shot_recall(triplet_rank, cls_matrix, obj_names, rel_name, root):
   
    train_data = read_json('train', root)
    scene_data = dict()
    for i in train_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            if str(rel[0]) not in objs.keys():
                #print(f'{rel[0]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            if str(rel[1]) not in objs.keys():
                #print(f'{rel[1]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                scene_data[triplet_name] = 1
            scene_data[triplet_name] += 1
    
    val_data = read_json('val', root)
    zero_shot_triplet = []
    count = 0
    for i in val_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            count += 1
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                zero_shot_triplet.append(triplet_name)
    
    # get valid triplet which not appears in train data
    valid_triplet = []
    non_zero_shot_triplet = []
    all_triplet = []

    for i in range(len(cls_matrix)):
        if cls_matrix[i, -1] == -1:
            continue
        if len(cls_matrix[i]) == 5:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][2]) + ' ' + str(cls_matrix[i][-1])
        elif len(cls_matrix[i]) == 3:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][1]) + ' ' + str(cls_matrix[i][-1])
        else:
            raise RuntimeError('unknown triplet length:', len(cls_matrix[i]))

        if triplet_name in zero_shot_triplet:
            valid_triplet.append(triplet_rank[i])
        else:
            non_zero_shot_triplet.append(triplet_rank[i])
        
        all_triplet.append(triplet_rank[i])
    
    valid_triplet = np.array(valid_triplet)
    non_zero_shot_triplet = np.array(non_zero_shot_triplet)
    all_triplet = np.array(all_triplet)

    zero_shot_50 = (valid_triplet <= 50).mean() * 100
    zero_shot_100 = (valid_triplet <= 100).mean() * 100

    non_zero_shot_50 = (non_zero_shot_triplet <= 50).mean() * 100
    non_zero_shot_100 = (non_zero_shot_triplet <= 100).mean() * 100

    all_50 = (all_triplet <= 50).mean() * 100
    all_100 = (all_triplet <= 100).mean() * 100

    return (zero_shot_50, zero_shot_100), (non_zero_shot_50, non_zero_shot_100), (all_50, all_100)

    

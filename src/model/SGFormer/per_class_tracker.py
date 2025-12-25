import torch
import numpy as np

class PerClassAccuracyTracker:
    def __init__(self, num_rel_classes=26, num_obj_classes=160):
        self.num_rel_classes = num_rel_classes
        self.num_obj_classes = num_obj_classes
        self.reset()
    
    def reset(self):
        # 关系类别的统计
        self.rel_stats = {
            i: {
                'support': 0,            # 真实正样本数
                'correct_topk': 0,       # 召回的正样本数
                'correct_top3': 0,       # Recall@3: 在top-3中被召回的正样本数
                'correct_top5': 0,       # Recall@5: 在top-5中被召回的正样本数
                'predicted_count': 0,    # 被预测的次数
                'correct_predictions': 0, # 正确预测的次数
            } for i in range(self.num_rel_classes)
        }
        
        # 物体类别的统计（如果指定了物体类别数量）
        if self.num_obj_classes is not None:
            self.obj_stats = {
                i: {
                    'support': 0,            # 真实出现次数
                    'correct_top1': 0,       # top-1预测正确的次数
                    'correct_topk': 0,       # top-k预测正确的次数
                    'correct_top3': 0,       # Recall@3: 在top-3中被召回的次数
                    'correct_top5': 0,       # Recall@5: 在top-5中被召回的次数
                    'total_samples': 0,      # 总样本数
                } for i in range(self.num_obj_classes)
            }
        else:
            self.obj_stats = {}
        
        # 总体统计
        self.total_rel_samples = 0
        self.correct_rel_top1 = 0
        self.correct_rel_topk = 0
        
        self.total_obj_samples = 0
        self.correct_obj_top1 = 0
        self.correct_obj_topk = 0
    
    def update_rel_batch(self, rels_preds, gt_edges, topk=1):
        """
        更新关系batch统计信息
        """
        for rel_idx, rel_pred in enumerate(rels_preds):
            self.total_rel_samples += 1
            
            # 获取预测结果
            sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
            gt_target = gt_edges[rel_idx][2] if len(gt_edges[rel_idx]) > 2 else []
            
            # 过滤有效的关系类别
            valid_targets = [t for t in gt_target if 0 <= t < self.num_rel_classes]
            top1_prediction = sorted_idx[0].item() if 0 <= sorted_idx[0].item() < self.num_rel_classes else -1
            topk_predictions = [idx for idx in sorted_idx[:topk].tolist() if 0 <= idx < self.num_rel_classes]
            
            # 计算整体准确率
            if len(valid_targets) == 0:
                # 无真实关系时的处理
                if top1_prediction == -1:  # 预测为无效类别（相当于None）
                    self.correct_rel_top1 += 1
                if len([p for p in topk_predictions if p >= 0]) == 0:  # top-k中也没有有效类别
                    self.correct_rel_topk += 1
            else:
                # 有真实关系
                if top1_prediction in valid_targets:
                    self.correct_rel_top1 += 1
                if len(set(topk_predictions) & set(valid_targets)) > 0:
                    self.correct_rel_topk += 1
            
            # 计算top-3和top-5预测
            top3_predictions = [idx for idx in sorted_idx[:3].tolist() if 0 <= idx < self.num_rel_classes]
            top5_predictions = [idx for idx in sorted_idx[:5].tolist() if 0 <= idx < self.num_rel_classes]
            
            # 更新每个类别的统计
            for class_id in range(self.num_rel_classes):
                # 支持数（真实正样本）
                if class_id in valid_targets:
                    self.rel_stats[class_id]['support'] += 1
                
                # 预测计数
                if class_id in topk_predictions:
                    self.rel_stats[class_id]['predicted_count'] += 1
                    # 正确预测（预测为该类且该类在真实标签中）
                    if class_id in valid_targets:
                        self.rel_stats[class_id]['correct_predictions'] += 1
                
                # 召回统计（真实为该类且被召回）
                if class_id in valid_targets:
                    if class_id in topk_predictions:
                        self.rel_stats[class_id]['correct_topk'] += 1
                    if class_id in top3_predictions:
                        self.rel_stats[class_id]['correct_top3'] += 1
                    if class_id in top5_predictions:
                        self.rel_stats[class_id]['correct_top5'] += 1
    
    def update_obj_batch(self, obj_preds, gt_cls, topk=1):
        """
        更新物体batch统计信息 (单标签分类)
        """
        if self.num_obj_classes is None:
            # 自动推断类别数量
            self.num_obj_classes = obj_preds.shape[1] if len(obj_preds) > 0 else 0
            self.obj_stats = {
                i: {
                    'support': 0,
                    'correct_top1': 0,
                    'correct_topk': 0,
                    'correct_top3': 0,
                    'correct_top5': 0,
                    'total_samples': 0,
                } for i in range(self.num_obj_classes)
            }
        
        for obj_idx, (obj_pred, gt_class) in enumerate(zip(obj_preds, gt_cls)):
            self.total_obj_samples += 1
            gt_class = gt_class.item() if isinstance(gt_class, torch.Tensor) else gt_class
            
            # 获取预测结果
            sorted_conf_matrix, sorted_idx = torch.sort(obj_pred, descending=True)
            top1_prediction = sorted_idx[0].item()
            topk_predictions = sorted_idx[:topk].tolist()
            
            # 计算准确率
            if top1_prediction == gt_class:
                self.correct_obj_top1 += 1
            
            if gt_class in topk_predictions:
                self.correct_obj_topk += 1
            
            # 计算top-3和top-5预测
            top3_predictions = sorted_idx[:3].tolist()
            top5_predictions = sorted_idx[:5].tolist()
            
            # 更新每个类别的统计
            for class_id in range(self.num_obj_classes):
                self.obj_stats[class_id]['total_samples'] += 1
                
                # 支持数（真实标签统计）
                if class_id == gt_class:
                    self.obj_stats[class_id]['support'] += 1
                    
                    # 预测正确统计
                    if top1_prediction == gt_class:
                        self.obj_stats[class_id]['correct_top1'] += 1
                    
                    if gt_class in topk_predictions:
                        self.obj_stats[class_id]['correct_topk'] += 1
                    
                    # Recall@3和Recall@5统计
                    if gt_class in top3_predictions:
                        self.obj_stats[class_id]['correct_top3'] += 1
                    if gt_class in top5_predictions:
                        self.obj_stats[class_id]['correct_top5'] += 1
    
    def get_rel_per_class_metrics(self):
        """
        计算关系每个类别的最终指标
        """
        results = {}
        for class_id in range(self.num_rel_classes):
            stats = self.rel_stats[class_id]
            
            # 精确率：正确预测为该类的样本 / 所有预测为该类的样本
            precision = stats['correct_predictions'] / stats['predicted_count'] if stats['predicted_count'] > 0 else 0
            
            # 召回率：正确召回的该类样本 / 该类的总正样本数
            recall = stats['correct_topk'] / stats['support'] if stats['support'] > 0 else 0
            
            # Recall@3: 在top-3中被召回的该类样本 / 该类的总正样本数
            recall_at_3 = stats['correct_top3'] / stats['support'] if stats['support'] > 0 else 0
            
            # Recall@5: 在top-5中被召回的该类样本 / 该类的总正样本数
            recall_at_5 = stats['correct_top5'] / stats['support'] if stats['support'] > 0 else 0
            
            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[class_id] = {
                'precision': precision,
                'recall': recall,
                'recall_at_3': recall_at_3,
                'recall_at_5': recall_at_5,
                'f1_score': f1,
                'support': stats['support'],
                'predicted_count': stats['predicted_count'],
                'correct_predictions': stats['correct_predictions'],
                'correct_topk': stats['correct_topk'],
                'correct_top3': stats['correct_top3'],
                'correct_top5': stats['correct_top5']
            }
        
        return results
    
    def get_obj_per_class_metrics(self):
        """
        计算物体每个类别的准确率
        """
        if self.num_obj_classes is None or self.num_obj_classes == 0:
            return {}
        
        results = {}
        for class_id in range(self.num_obj_classes):
            stats = self.obj_stats[class_id]
            
            # Top-1准确率：该类别预测正确的样本数 / 总样本数
            top1_acc = stats['correct_top1'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
            
            # Top-k准确率：该类别在top-k中被预测到的样本数 / 总样本数
            topk_acc = stats['correct_topk'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
            
            # 召回率（对于单标签分类，这表示在该类别为真实标签时的召回率）
            recall = stats['correct_topk'] / stats['support'] if stats['support'] > 0 else 0
            
            # Recall@3: 在该类别为真实标签时，在top-3中被召回的次数 / 该类的总正样本数
            recall_at_3 = stats['correct_top3'] / stats['support'] if stats['support'] > 0 else 0
            
            # Recall@5: 在该类别为真实标签时，在top-5中被召回的次数 / 该类的总正样本数
            recall_at_5 = stats['correct_top5'] / stats['support'] if stats['support'] > 0 else 0
            
            results[class_id] = {
                'class_accuracy': top1_acc,      # 每个类别的准确率
                'topk_accuracy': topk_acc,
                'recall': recall,
                'recall_at_3': recall_at_3,
                'recall_at_5': recall_at_5,
                'support': stats['support'],
                'total_samples': stats['total_samples'],
                'correct_top1': stats['correct_top1'],
                'correct_topk': stats['correct_topk'],
                'correct_top3': stats['correct_top3'],
                'correct_top5': stats['correct_top5']
            }
        
        return results
    
    def get_overall_accuracy(self):
        """
        获取整体准确率
        """
        rel_top1_acc = self.correct_rel_top1 / self.total_rel_samples if self.total_rel_samples > 0 else 0
        rel_topk_acc = self.correct_rel_topk / self.total_rel_samples if self.total_rel_samples > 0 else 0
        
        obj_top1_acc = self.correct_obj_top1 / self.total_obj_samples if self.total_obj_samples > 0 else 0
        obj_topk_acc = self.correct_obj_topk / self.total_obj_samples if self.total_obj_samples > 0 else 0
        
        return {
            'rel': {
                'top1_accuracy': rel_top1_acc,
                'topk_accuracy': rel_topk_acc,
                'total_samples': self.total_rel_samples,
                'correct_top1': self.correct_rel_top1,
                'correct_topk': self.correct_rel_topk
            },
            'obj': {
                'top1_accuracy': obj_top1_acc,
                'topk_accuracy': obj_topk_acc,
                'total_samples': self.total_obj_samples,
                'correct_top1': self.correct_obj_top1,
                'correct_topk': self.correct_obj_topk
            }
        }
    
    def print_comprehensive_results(self, relation_names=None, obj_names=None, rel_topk=1, obj_topk=1):
        """
        打印综合结果
        """
        overall_metrics = self.get_overall_accuracy()
        
        # 打印关系结果
        if self.total_rel_samples > 0:
            if relation_names is None:
                relation_names = [f'Relation_{i}' for i in range(self.num_rel_classes)]
            
            print(f"Relationship Overall Accuracy Analysis")
            print("=" * 50)
            print(f"Total Samples: {overall_metrics['rel']['total_samples']}")
            print(f"Top-1 Accuracy: {overall_metrics['rel']['top1_accuracy']:.4f}")
            print(f"Top-{rel_topk} Accuracy: {overall_metrics['rel']['topk_accuracy']:.4f}")
            print()
            
            # 打印每个关系类别的详细结果
            rel_metrics = self.get_rel_per_class_metrics()
            print(f"Per-Class Relationship Detailed Analysis")
            print("=" * 150)
            print(f"{'Relation':<25} {'Precision':<10} {'Recall':<10} {'Recall@3':<10} {'Recall@5':<10} {'F1-Score':<10} "
                  f"{'Support':<8} {'Pred_Count':<12} {'Correct':<8}")
            print("-" * 150)
            
            sorted_metrics = sorted(rel_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            
            for class_id, class_metrics in sorted_metrics:
                name = relation_names[class_id] if class_id < len(relation_names) else f"Relation_{class_id}"
                
                print(f"{name:<25} "
                      f"{class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} "
                      f"{class_metrics['recall_at_3']:<10.4f} "
                      f"{class_metrics['recall_at_5']:<10.4f} "
                      f"{class_metrics['f1_score']:<10.4f} "
                      f"{class_metrics['support']:<8} "
                      f"{class_metrics['predicted_count']:<12} "
                      f"{class_metrics['correct_predictions']:<8}")
            
            print("-" * 150)
            
            # 计算关系平均指标
            valid_classes = [m for m in rel_metrics.values() if m['support'] > 0]
            if valid_classes:
                mean_precision = np.mean([m['precision'] for m in valid_classes])
                mean_recall = np.mean([m['recall'] for m in valid_classes])
                mean_recall_at_3 = np.mean([m['recall_at_3'] for m in valid_classes])
                mean_recall_at_5 = np.mean([m['recall_at_5'] for m in valid_classes])
                mean_f1 = np.mean([m['f1_score'] for m in valid_classes])
                
                print(f"Mean Precision: {mean_precision:.4f}")
                print(f"Mean Recall: {mean_recall:.4f}")
                print(f"Mean Recall@3: {mean_recall_at_3:.4f}")
                print(f"Mean Recall@5: {mean_recall_at_5:.4f}")
                print(f"Mean F1-Score: {mean_f1:.4f}")
            print("\n")
        
        # 打印物体结果
        if self.total_obj_samples > 0 and self.num_obj_classes is not None:
            if obj_names is None:
                obj_names = [f'Object_{i}' for i in range(self.num_obj_classes)]
            
            print(f"Object Overall Accuracy Analysis")
            print("=" * 50)
            print(f"Total Samples: {overall_metrics['obj']['total_samples']}")
            print(f"Top-1 Accuracy: {overall_metrics['obj']['top1_accuracy']:.4f}")
            print(f"Top-{obj_topk} Accuracy: {overall_metrics['obj']['topk_accuracy']:.4f}")
            print()
            
            # 打印每个物体类别的准确率
            obj_metrics = self.get_obj_per_class_metrics()
            print(f"Per-Class Object Accuracy Analysis")
            print("=" * 110)
            print(f"{'Object':<30} {'Accuracy':<10} {'Top-K Acc':<10} {'Recall':<10} {'Recall@3':<10} {'Recall@5':<10} {'Support':<8}")
            print("-" * 110)
            
            # 按准确率排序
            sorted_obj_metrics = sorted(obj_metrics.items(), key=lambda x: x[1]['class_accuracy'], reverse=True)
            
            for class_id, class_metrics in sorted_obj_metrics:
                name = obj_names[class_id] if class_id < len(obj_names) else f"Object_{class_id}"
                
                print(f"{name:<30} "
                      f"{class_metrics['class_accuracy']:<10.4f} "
                      f"{class_metrics['topk_accuracy']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} "
                      f"{class_metrics['recall_at_3']:<10.4f} "
                      f"{class_metrics['recall_at_5']:<10.4f} "
                      f"{class_metrics['support']:<8}")
            
            print("-" * 110)
            
            # 计算物体平均准确率
            valid_obj_classes = [m for m in obj_metrics.values() if m['support'] > 0]
            if valid_obj_classes:
                mean_obj_acc = np.mean([m['class_accuracy'] for m in valid_obj_classes])
                mean_obj_topk_acc = np.mean([m['topk_accuracy'] for m in valid_obj_classes])
                mean_obj_recall = np.mean([m['recall'] for m in valid_obj_classes])
                mean_obj_recall_at_3 = np.mean([m['recall_at_3'] for m in valid_obj_classes])
                mean_obj_recall_at_5 = np.mean([m['recall_at_5'] for m in valid_obj_classes])
                
                print(f"Mean Object Accuracy: {mean_obj_acc:.4f}")
                print(f"Mean Object Top-{obj_topk} Accuracy: {mean_obj_topk_acc:.4f}")
                print(f"Mean Object Recall: {mean_obj_recall:.4f}")
                print(f"Mean Object Recall@3: {mean_obj_recall_at_3:.4f}")
                print(f"Mean Object Recall@5: {mean_obj_recall_at_5:.4f}")
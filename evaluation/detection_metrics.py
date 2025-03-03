# evaluation/detection_metrics.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from perception.utils.transforms import calculate_iou

def mean_average_precision(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching predictions to ground truth
        num_classes: Number of classes (if None, inferred from data)
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    # Extract class IDs from ground truth
    gt_classes = set()
    for gt in ground_truth:
        gt_classes.add(gt['class_id'])
    
    if num_classes is None:
        num_classes = max(gt_classes) + 1
    
    # Initialize per-class metrics
    average_precisions = {}
    
    # Calculate AP for each class
    for class_id in range(num_classes):
        if class_id not in gt_classes:
            continue
        
        # Get predictions and ground truth for this class
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_gt = [g for g in ground_truth if g['class_id'] == class_id]
        
        # If no predictions or no ground truth, AP is 0
        if not class_preds or not class_gt:
            average_precisions[class_id] = 0.0
            continue
        
        # Sort predictions by confidence score (descending)
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
        
        # Create arrays for precision-recall calculation
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        # Mark ground truth as unmatched initially
        gt_matched = np.zeros(len(class_gt), dtype=bool)
        
        # Process each prediction
        for i, pred in enumerate(class_preds):
            # Find the best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gt):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if the match is valid
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        recall = cumsum_tp / len(class_gt)
        
        # Compute Average Precision using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        average_precisions[class_id] = ap
    
    # Calculate mAP
    mAP = np.mean(list(average_precisions.values())) if average_precisions else 0.0
    
    # Prepare result
    result = {
        'mAP': mAP,
        'AP': average_precisions
    }
    
    return result


def precision_recall_curve(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_id: int,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve for a specific class.
    
    Args:
        predictions: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold for matching predictions to ground truth
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Get predictions and ground truth for this class
    class_preds = [p for p in predictions if p['class_id'] == class_id]
    class_gt = [g for g in ground_truth if g['class_id'] == class_id]
    
    # If no predictions or no ground truth, return empty arrays
    if not class_preds or not class_gt:
        return np.array([]), np.array([]), np.array([])
    
    # Sort predictions by confidence score (descending)
    class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
    
    # Create arrays for precision-recall calculation
    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    thresholds = np.array([pred['score'] for pred in class_preds])
    
    # Mark ground truth as unmatched initially
    gt_matched = np.zeros(len(class_gt), dtype=bool)
    
    # Process each prediction
    for i, pred in enumerate(class_preds):
        # Find the best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(class_gt):
            if gt_matched[j]:
                continue
            
            iou = calculate_iou(pred['box'], gt['box'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if the match is valid
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute cumulative precision and recall
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    
    precision = cumsum_tp / (cumsum_tp + cumsum_fp)
    recall = cumsum_tp / len(class_gt)
    
    return precision, recall, thresholds


def detection_statistics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate comprehensive detection statistics.
    
    Args:
        predictions: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching predictions to ground truth
        
    Returns:
        Dictionary with detection statistics
    """
    # Count total predictions and ground truth
    num_predictions = len(predictions)
    num_ground_truth = len(ground_truth)
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Mark ground truth as unmatched initially
    gt_matched = np.zeros(num_ground_truth, dtype=bool)
    
    # Sort predictions by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Match predictions to ground truth
    for pred in predictions:
        # Find the best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            
            # Skip if class doesn't match (for multi-class evaluation)
            if 'class_id' in pred and 'class_id' in gt and pred['class_id'] != gt['class_id']:
                continue
            
            iou = calculate_iou(pred['box'], gt['box'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if the match is valid
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            true_positives += 1
            gt_matched[best_gt_idx] = True
        else:
            false_positives += 1
    
    # Unmatched ground truth are false negatives
    false_negatives = num_ground_truth - true_positives
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare result
    result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'num_predictions': num_predictions,
        'num_ground_truth': num_ground_truth,
        'iou_threshold': iou_threshold
    }
    
    return result



# evaluation/tracking_metrics.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict

def calculate_tracking_metrics(
    tracks_gt: List[Dict],
    tracks_pred: List[Dict],
    iou_threshold: float = 0.5,
    max_switch_cost: float = 0.8
) -> Dict[str, float]:
    """
    Calculate multi-object tracking metrics (MOTA, MOTP, ID switches, etc.).
    
    Args:
        tracks_gt: List of ground truth tracks by frame
                  Each item is {frame_id, track_id, box}
        tracks_pred: List of predicted tracks by frame
                    Each item is {frame_id, track_id, box}
        iou_threshold: IoU threshold for matching predictions to ground truth
        max_switch_cost: Maximum cost for considering ID switches
        
    Returns:
        Dictionary with tracking metrics
    """
    # Group tracks by frame
    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    
    for gt in tracks_gt:
        gt_by_frame[gt['frame_id']].append(gt)
    
    for pred in tracks_pred:
        pred_by_frame[pred['frame_id']].append(pred)
    
    # Find all frames and track IDs
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    
    gt_ids = set()
    for tracks in gt_by_frame.values():
        for track in tracks:
            gt_ids.add(track['track_id'])
    
    # Initialize metrics
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_matches = 0
    total_iou = 0.0
    
    # Track assignments from previous frame
    prev_id_map = {}  # gt_id -> pred_id
    
    # Process each frame
    for frame_id in all_frames:
        gt_tracks = gt_by_frame[frame_id]
        pred_tracks = pred_by_frame[frame_id]
        
        # Count ground truth in this frame
        total_gt += len(gt_tracks)
        
        # If no ground truth, all predictions are false positives
        if not gt_tracks:
            total_fp += len(pred_tracks)
            continue
        
        # If no predictions, all ground truth are false negatives
        if not pred_tracks:
            total_fn += len(gt_tracks)
            continue
        
        # Build cost matrix based on IOU
        cost_matrix = np.ones((len(gt_tracks), len(pred_tracks)))
        
        for i, gt in enumerate(gt_tracks):
            for j, pred in enumerate(pred_tracks):
                iou = calculate_iou_from_dicts(gt, pred)
                if iou >= iou_threshold:
                    cost_matrix[i, j] = 1.0 - iou
                else:
                    cost_matrix[i, j] = float('inf')
        
        # Find assignments
        from scipy.optimize import linear_sum_assignment
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Count matches, false positives, and false negatives
        matches = []
        for i, j in zip(gt_indices, pred_indices):
            if cost_matrix[i, j] != float('inf'):
                matches.append((i, j))
                
                # Compute IOU for matched pairs
                iou = 1.0 - cost_matrix[i, j]
                total_iou += iou
        
        total_matches += len(matches)
        total_fp += len(pred_tracks) - len(matches)
        total_fn += len(gt_tracks) - len(matches)
        
        # Count ID switches
        curr_id_map = {}
        for i, j in matches:
            gt_id = gt_tracks[i]['track_id']
            pred_id = pred_tracks[j]['track_id']
            
            curr_id_map[gt_id] = pred_id
            
            # Check for ID switch
            if gt_id in prev_id_map and prev_id_map[gt_id] != pred_id:
                total_id_switches += 1
        
        # Update previous assignments
        prev_id_map = curr_id_map
    
    # Calculate final metrics
    mota = 1.0 - (total_fp + total_fn + total_id_switches) / total_gt if total_gt > 0 else 0.0
    motp = total_iou / total_matches if total_matches > 0 else 0.0
    
    result = {
        'MOTA': mota,  # Multiple Object Tracking Accuracy
        'MOTP': motp,  # Multiple Object Tracking Precision
        'ID_Switches': total_id_switches,
        'Matches': total_matches,
        'FP': total_fp,  # False Positives
        'FN': total_fn,  # False Negatives
        'GT': total_gt,  # Total Ground Truth
        'Precision': total_matches / (total_matches + total_fp) if (total_matches + total_fp) > 0 else 0.0,
        'Recall': total_matches / (total_matches + total_fn) if (total_matches + total_fn) > 0 else 0.0
    }
    
    return result


def calculate_track_statistics(
    tracks_gt: List[Dict],
    tracks_pred: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-track statistics.
    
    Args:
        tracks_gt: List of ground truth tracks by frame
        tracks_pred: List of predicted tracks by frame
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with per-track statistics
    """
    # Group tracks by ID
    gt_by_id = defaultdict(list)
    pred_by_id = defaultdict(list)
    
    for gt in tracks_gt:
        gt_by_id[gt['track_id']].append(gt)
    
    for pred in tracks_pred:
        pred_by_id[pred['track_id']].append(pred)
    
    # Calculate track statistics
    track_stats = {}
    
    # Match predicted tracks to ground truth tracks
    gt_matches = {}  # gt_id -> pred_id
    
    # For each ground truth track
    for gt_id, gt_frames in gt_by_id.items():
        # Count matches with each predicted track
        pred_matches = defaultdict(int)
        
        for gt_frame in gt_frames:
            frame_id = gt_frame['frame_id']
            
            # Find predictions in the same frame
            frame_preds = [p for p in tracks_pred if p['frame_id'] == frame_id]
            
            # Find best matching prediction
            best_iou = 0.0
            best_pred_id = None
            
            for pred in frame_preds:
                iou = calculate_iou_from_dicts(gt_frame, pred)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_id = pred['track_id']
            
            if best_pred_id is not None:
                pred_matches[best_pred_id] += 1
        
        # Assign best matching predicted track
        if pred_matches:
            best_pred_id = max(pred_matches.keys(), key=lambda k: pred_matches[k])
            match_ratio = pred_matches[best_pred_id] / len(gt_frames)
            
            if match_ratio > 0.5:  # More than half of frames match
                gt_matches[gt_id] = best_pred_id
    
    # Calculate statistics for each ground truth track
    for gt_id, gt_frames in gt_by_id.items():
        track_len = len(gt_frames)
        track_duration = max(f['frame_id'] for f in gt_frames) - min(f['frame_id'] for f in gt_frames) + 1
        
        stats = {
            'length': track_len,
            'duration': track_duration,
            'completeness': track_len / track_duration,
            'matched': gt_id in gt_matches
        }
        
        if gt_id in gt_matches:
            pred_id = gt_matches[gt_id]
            pred_frames = pred_by_id[pred_id]
            
            # Count correct matches
            correct_matches = 0
            total_iou = 0.0
            
            for gt_frame in gt_frames:
                frame_id = gt_frame['frame_id']
                
                # Find matching prediction frame
                matching_preds = [p for p in pred_frames if p['frame_id'] == frame_id]
                
                if matching_preds:
                    pred_frame = matching_preds[0]  # Should be only one
                    iou = calculate_iou_from_dicts(gt_frame, pred_frame)
                    
                    if iou >= iou_threshold:
                        correct_matches += 1
                        total_iou += iou
            
            # Calculate track-level metrics
            track_precision = correct_matches / len(pred_frames) if pred_frames else 0.0
            track_recall = correct_matches / track_len
            
            stats.update({
                'pred_id': pred_id,
                'pred_length': len(pred_frames),
                'matches': correct_matches,
                'track_precision': track_precision,
                'track_recall': track_recall,
                'avg_iou': total_iou / correct_matches if correct_matches > 0 else 0.0
            })
        
        track_stats[gt_id] = stats
    
    return track_stats


def calculate_iou_from_dicts(box1: Dict, box2: Dict) -> float:
    """
    Calculate IoU between two bounding boxes stored in dictionaries.
    
    Args:
        box1: First box dictionary with 'box' field
        box2: Second box dictionary with 'box' field
        
    Returns:
        IoU score (0-1)
    """
    # Extract boxes
    if 'box' in box1 and 'box' in box2:
        return calculate_iou(box1['box'], box2['box'])
    else:
        return 0.0


def track_fragmentation(tracks: List[Dict]) -> Dict[int, int]:
    """
    Calculate track fragmentation (number of separate tracks per object).
    
    Args:
        tracks: List of track dictionaries with object_id and track_id
        
    Returns:
        Dictionary mapping object_id to fragment count
    """
    # Count unique track IDs per object
    fragments = defaultdict(set)
    
    for track in tracks:
        if 'object_id' in track and 'track_id' in track:
            fragments[track['object_id']].add(track['track_id'])
    
    # Convert sets to counts
    return {obj_id: len(track_ids) for obj_id, track_ids in fragments.items()}


def mostly_tracked_mostly_lost(
    tracks_gt: List[Dict],
    tracks_pred: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate mostly tracked (MT) and mostly lost (ML) metrics.
    
    Args:
        tracks_gt: List of ground truth tracks by frame
        tracks_pred: List of predicted tracks by frame
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with MT and ML metrics
    """
    # Calculate track statistics
    track_stats = calculate_track_statistics(tracks_gt, tracks_pred, iou_threshold)
    
    # Count track categories
    total_tracks = len(track_stats)
    
    mostly_tracked = 0  # Tracked for at least 80% of duration
    partially_tracked = 0  # Tracked between 20% and 80% of duration
    mostly_lost = 0  # Tracked for less than 20% of duration
    
    for _, stats in track_stats.items():
        if not stats['matched']:
            mostly_lost += 1
        else:
            track_recall = stats['track_recall']
            
            if track_recall >= 0.8:
                mostly_tracked += 1
            elif track_recall >= 0.2:
                partially_tracked += 1
            else:
                mostly_lost += 1
    
    # Calculate percentages
    mt_percentage = mostly_tracked / total_tracks if total_tracks > 0 else 0.0
    pt_percentage = partially_tracked / total_tracks if total_tracks > 0 else 0.0
    ml_percentage = mostly_lost / total_tracks if total_tracks > 0 else 0.0
    
    result = {
        'MT': mostly_tracked,
        'PT': partially_tracked,
        'ML': mostly_lost,
        'MT_percentage': mt_percentage,
        'PT_percentage': pt_percentage,
        'ML_percentage': ml_percentage,
        'total_tracks': total_tracks
    }
    
    return result
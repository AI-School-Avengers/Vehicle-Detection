import torch

def iou(boxes_preds, boxes_labels):
    # boxes_preds shape : (N,4) N은 bboxes의 갯수
    # boxes_labels shape : (N,4)
    box1_x1 = boxes_preds[..., 0:1] # (N,1)
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # .clamp(0)은 intersect하지 않는 경우 / 최소값을 0으로 설정

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, score_threshold, iou_threshold):
    print("="*40 + "nms Start" + "="*40)
    result_boxes = []

    if type(bboxes) != list:     # bboxes가 list인지 확인
        raise Exception("bboxes type is not list")

    bboxes = [bbox for bbox in bboxes if bbox['score'] > score_threshold]   # 1) score가 임계값 이하인 bbox 제거
    bboxes = sorted(bboxes,key = lambda x : x['score'], reverse= True)      # 2) score을 기준으로 내림차순 정렬: sorted(정렬할 데이터, lambda x : 정렬기준, revers = False (defalt는 오름차순))
    boxes = []

    while bboxes:                                                           # 3) iou가 임계치 이상이면서 hosen_box와 class명이 같은 box는 제거(동일 물체를 detect했다고 판정)
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box['label'] != chosen_box['label'] \
                  or iou(torch.tensor(chosen_box['bbox']),torch.tensor(box['bbox']))
                  < iou_threshold]
        result_boxes.append(chosen_box)

    return  result_boxes


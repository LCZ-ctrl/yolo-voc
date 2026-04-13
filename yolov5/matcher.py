import torch
import numpy as np


class Yolov5Matcher(object):
    def __init__(self, num_classes, num_anchors, anchor_size, anchor_threshold, iou_thresh):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_threshold = anchor_threshold
        self.iou_thresh = iou_thresh
        self.anchor_sizes = np.array(
            [[anchor[0], anchor[1]]
             for anchor in anchor_size]
        )
        self.anchor_boxes = np.array(
            [[0., 0., anchor[0], anchor[1]]
             for anchor in anchor_size]
        )  # [KA, 4]

    def compute_iou(self, anchor_boxes, gt_box):
        """
        Compute IoU between ground truth box and five anchor boxes
        """
        # anchors: [K*A, 4]
        anchors = np.zeros_like(anchor_boxes)
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]

        # gt_box: [1, 4] -> [K*A, 4]
        gt_box = np.array(gt_box).reshape(-1, 4)
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # compute intersection
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h

        # compute union
        union_area = anchors_area + gt_box_area - inter_area

        # compute IoU
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)

        return iou  # [A,]

    def iou_assignment(self, ctr_points, gt_box, fpn_strides):
        # compute iou
        iou = self.compute_iou(self.anchor_boxes, gt_box)
        iou_mask = (iou > self.iou_thresh)

        # positive sample assignment according to IoU
        label_assignment_results = []
        if iou_mask.sum() == 0:
            # Case 1: all prior boxes have low IoU with the gt_box
            # mark the prior box with the highest IoU as positive sample

            # get index of prior box
            iou_ind = np.argmax(iou)

            # pyramid level
            level = iou_ind // self.num_anchors

            # anchor index
            anchor_idx = iou_ind - level * self.num_anchors

            # stride
            stride = fpn_strides[level]

            # calculate grid coordinate of center point
            xc, yc = ctr_points
            xc_s = xc / stride
            yc_s = yc / stride
            grid_x = int(xc_s)
            grid_y = int(yc_s)

            # save grid coordinate and anchor box index for positive sample
            label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])
        else:
            # Case 2&3: at least one prior box have enough IoU with the gt_box
            for iou_ind, iou_m in enumerate(iou_mask):
                if iou_m:
                    # pyramid level
                    level = iou_ind // self.num_anchors

                    # get index of prior box
                    anchor_idx = iou_ind - level * self.num_anchors

                    # stride
                    stride = fpn_strides[level]

                    # calculate grid coordinate of center point
                    xc, yc = ctr_points
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    # save grid coordinate and anchor box index for positive samples
                    label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])

        return label_assignment_results

    def aspect_ratio_assignment(self, ctr_points, keeps, fpn_strides):
        label_assignment_results = []
        for keep_idx, keep in enumerate(keeps):
            if keep:
                level = keep_idx // self.num_anchors  # pyramid level
                anchor_idx = keep_idx - level * self.num_anchors  # anchor index

                # get the corresponding stride
                stride = fpn_strides[level]

                # compute the gride cell
                xc, yc = ctr_points
                xc_s = xc / stride
                yc_s = yc / stride
                grid_x = int(xc_s)
                grid_y = int(yc_s)

                label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])

        return label_assignment_results

    @torch.no_grad()
    def __call__(self, fmp_sizes, fpn_strides, targets):
        assert len(fmp_sizes) == len(fpn_strides)

        bs = len(targets)
        gt_objectness = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1])
            for (fmp_h, fmp_w) in fmp_sizes
        ]
        gt_classes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes])
            for (fmp_h, fmp_w) in fmp_sizes
        ]
        gt_bboxes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4])
            for (fmp_h, fmp_w) in fmp_sizes
        ]

        # iterate through each image in the batch
        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            # iterate through the labels of each target in the image
            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # target bbox coordinate
                x1, y1, x2, y2 = gt_box.tolist()

                # calculate center point and width and height of the target bbox
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = np.array([[0., 0., bw, bh]])

                # check if the target bbox is valid
                if bw < 1.0 or bh < 1.0:
                    continue

                # compute aspect ratio
                ratios = gt_box[..., 2:] / self.anchor_sizes
                keeps = np.maximum(ratios, 1 / ratios).max(-1) < self.anchor_threshold

                if keeps.sum() == 0:
                    label_assignment_results = self.iou_assignment([xc, yc], gt_box, fpn_strides)
                else:
                    label_assignment_results = self.aspect_ratio_assignment([xc, yc], keeps, fpn_strides)

                # label assignment
                for result in label_assignment_results:
                    # assignment
                    grid_x, grid_y, xc_s, yc_s, level, anchor_idx = result
                    stride = fpn_strides[level]
                    fmp_h, fmp_w = fmp_sizes[level]
                    # coord on the feature
                    x1s, y1s = x1 / stride, y1 / stride
                    x2s, y2s = x2 / stride, y2 / stride
                    # offset
                    off_x = xc_s - grid_x
                    off_y = yc_s - grid_y

                    if off_x <= 0.5 and off_y <= 0.5:  # top left
                        grids = [(grid_x - 1, grid_y), (grid_x, grid_y - 1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y <= 0.5:  # top right
                        grids = [(grid_x + 1, grid_y), (grid_x, grid_y - 1), (grid_x, grid_y)]
                    elif off_x <= 0.5 and off_y > 0.5:  # bottom left
                        grids = [(grid_x - 1, grid_y), (grid_x, grid_y + 1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y > 0.5:  # bottom right
                        grids = [(grid_x + 1, grid_y), (grid_x, grid_y + 1), (grid_x, grid_y)]

                    for (i, j) in grids:
                        is_in_box = (j >= y1s and j < y2s) and (i >= x1s and i < x2s)
                        is_valid = (j >= 0 and j < fmp_h) and (i >= 0 and i < fmp_w)

                        if is_in_box and is_valid:
                            # obj
                            gt_objectness[level][batch_index, j, i, anchor_idx] = 1.0
                            # cls
                            cls_ont_hot = torch.zeros(self.num_classes)
                            cls_ont_hot[int(gt_label)] = 1.0
                            gt_classes[level][batch_index, j, i, anchor_idx] = cls_ont_hot
                            # box
                            gt_bboxes[level][batch_index, j, i, anchor_idx] = torch.as_tensor([x1, y1, x2, y2])

        # [B, M, C]
        gt_objectness = torch.cat([gt.view(bs, -1, 1) for gt in gt_objectness], dim=1).float()
        gt_classes = torch.cat([gt.view(bs, -1, self.num_classes) for gt in gt_classes], dim=1).float()
        gt_bboxes = torch.cat([gt.view(bs, -1, 4) for gt in gt_bboxes], dim=1).float()

        return gt_objectness, gt_classes, gt_bboxes

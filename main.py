import pickle
import torch
import iou3d_nms_utils
import numpy as np
import socket
import struct
import open3d
from visual_utils import open3d_vis_utils as V

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans

def decode_torch(box_encodings, anchors):
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    if not False:
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
    else:
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

    diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za

    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza

    if False:
        rg_cos = cost + torch.cos(ra)
        rg_sin = sint + torch.sin(ra)
        rg = torch.atan2(rg_sin, rg_cos)
    else:
        rg = rt + ra

    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

def generate_predicted_boxes(batch_size, cls_preds, box_preds, dir_cls_preds=None):
    anchors = np.load("anchors.npy")
    anchors = torch.from_numpy(anchors)
    cls_preds = torch.from_numpy(cls_preds)
    box_preds = torch.from_numpy(box_preds)
    dir_cls_preds = torch.from_numpy(dir_cls_preds)

    num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
    batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
    batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
        if not isinstance(cls_preds, list) else cls_preds
    batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
        else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
    batch_box_preds = decode_torch(batch_box_preds, batch_anchors)

    if dir_cls_preds is not None:
        dir_offset = 0.78539
        dir_limit_offset = 0
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
            else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        period = (2 * np.pi / 2)
        dir_rot = limit_period(
            batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
        )
        batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

    # if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
    #     batch_box_preds[..., 6] = limit_period(
    #         -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
    #     )
    #     # print("预测形状")
    #     # print(batch_cls_preds)
    #     # print(batch_box_preds)
    #     # 预测分类
    return batch_cls_preds, batch_box_preds   
        
def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    # print(box_scores)
    # print(score_thresh)
    nms_config = {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.01, 'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(4096, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, 'nms_gpu')(
                boxes_for_nms[:, 0:7], box_scores_nms, 0.01, **nms_config
        )
        selected = indices[keep_idx[:500]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]

def post_processing(batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = {'RECALL_THRESH_LIST': [0.3, 0.5, 0.7], 'SCORE_THRESH': 0.1, 'OUTPUT_RAW_SCORE': False, 'EVAL_METRIC': 'kitti', 'NMS_CONFIG': {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.01, 'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}}
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        num_class = 1
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, num_class]

                if not batch_dict['cls_preds_normalized']:
                    print("执行4")
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                print("执行5")
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    print("执行6")
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if False:
                print("执行7")
                if not isinstance(cls_preds, list):
                    print("执行8")
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, num_class, device=cls_preds[0].device)]
                else:
                    print("执行9")
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                print("执行10")
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    print("执行11")
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    print("执行12")
                    label_preds = label_preds + 1
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config='nms',
                    score_thresh=0.1
                )

                if False:
                    print("执行13")
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=[0.3, 0.5, 0.7]
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

if __name__ == '__main__':
    hostname = '172.20.107.96'
    port = 55555
    addr = (hostname, port)
    srv = socket.socket()
    srv.bind(addr)
    srv.listen(5)
    print("Server waitting for connection")
    # connect_socket, client_addr = srv.accept()
    # print("Connected client: ", client_addr)
    post_i = 0

    connect_socket, client_addr = srv.accept()
    print("Connected client: ", client_addr)
    rec_size = connect_socket.recv(1024)
    size = struct.unpack('i', rec_size)[0]
    connect_socket.send(bytes("send data", encoding='utf-8'))


    rec_data = b''
    while (len(rec_data)) < size:
        rec_data += connect_socket.recv(1024)
    np_array = pickle.loads(rec_data)
    input_0 = np_array.reshape(-1,4)
    print(np_array.size)
    # np.save('data/000000.npy',np_array)
    connect_socket.send(bytes("ok", encoding='utf-8'))
    connect_socket.close()
    input_1 = input_0
    while True:
        input_0 = input_1
        bin_idex = '%06d' % (post_i + 1)

        connect_socket, client_addr = srv.accept()
        print("Connected client: ", client_addr)
        rec_size = connect_socket.recv(1024)
        print("size")
        size = struct.unpack('i', rec_size)[0]
        connect_socket.send(bytes("send data", encoding='utf-8'))
        
        
        rec_data = b''
        while (len(rec_data)) < size:
            rec_data += connect_socket.recv(1024)
        np_array = pickle.loads(rec_data)
        input_1 = np_array.reshape(-1,4)
        print(np_array.size)
        # np.save('data/'+str(bin_idex)+'.npy',np_array)
        connect_socket.send(bytes("ok", encoding='utf-8'))
        connect_socket.close()
        print("input close")

        connect_socket, client_addr = srv.accept()
        print("Connected client: ", client_addr)

        image_idex = '%06d' % post_i
        rec_data = b''
        while True:
            rec = connect_socket.recv(1024)
            rec_data += rec
            if(len(rec_data)==65702):
                break
        np_array1 = pickle.loads(rec_data)
        cls_preds = np_array1.reshape(1,64,128,2)
        print(np_array1.size)
        # np.save('result/'+str(image_idex)+'_cls_preds.npy',np_array1)
        connect_socket.send(bytes("ok", encoding='utf-8'))
        
        rec_data = b''
        while True:
            rec = connect_socket.recv(1024)
            rec_data += rec
            if(len(rec_data)==458918):
                break      
        np_array2 = pickle.loads(rec_data)
        box_preds = np_array2.reshape(1,64,128,14)
        print(np_array2.size)
        # np.save('result/'+str(image_idex)+'_box_preds.npy',np_array2)
        connect_socket.send(bytes("ok", encoding='utf-8'))
        
        rec_data = b''
        while True:
            rec = connect_socket.recv(1024)
            rec_data += rec
            if(len(rec_data)==131238):
                break      
        np_array3 = pickle.loads(rec_data)
        dir_cls_preds = np_array3.reshape(1,64,128,4)
        print(np_array3.size)
        # np.save('result/'+str(image_idex)+'_dir_cls_preds.npy',np_array3)
        connect_socket.send(bytes("ok", encoding='utf-8'))
        connect_socket.close()
        
        post_i += 1

        # cls_preds = np.load("/home/star/scp/OpenPCDet_Project/tools/cls_preds116.npy")
        # print(cls_preds.shape)
        # box_preds = np.load("/home/star/scp/OpenPCDet_Project/tools/box_preds116.npy")
        # dir_cls_preds = np.load("/home/star/scp/OpenPCDet_Project/tools/dir_cls_preds116.npy")
        batch_cls_preds, batch_box_preds = generate_predicted_boxes(
            batch_size=1,
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )

        fp = open("batch_dict.pkl","rb+")
        s = pickle.load(fp)#序列化打印结果
        s['batch_cls_preds'] = batch_cls_preds.cuda()
        s['batch_box_preds'] = batch_box_preds.cuda()
        # s['cls_preds_normalized'] = False
        # print(s)
        pred_dicts, recall_dict = post_processing(s)
        print(input_0)

        V.draw_scenes(
                points=input_0, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
        print(pred_dicts)
        print("hello world")
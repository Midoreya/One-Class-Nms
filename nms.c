//---------------------//
#include <stdio.h>
#include <stdlib.h>
//---------------------//
#include "nms.h"
//---------------------//

int create_xc(float *prediction, int *xc, float conf_thres, int len_total,
              int len_group) {
  int i = 0;
  int total_num = 0;

  for (i = 0; i < len_total; i++) {
    if (prediction[i * len_group + 4] > conf_thres) {
      xc[i] = 1;
      total_num++;
    } else
      xc[i] = 0;
  }

  if (total_num == 0) printf("No items\n");

  return total_num;
}

int *create_xc_index(int *xc_index, int total_num, int *xc, int len_total) {
  int i = 0, j = 0;
  xc_index = (int *)calloc(total_num, sizeof(int));

  for (i = 0; i < len_total; i++) {
    if (xc[i] == 1) {
      xc_index[j] = i;
      j++;
    }
  }

  return xc_index;
}

int create_dets(detection *dets, int total, int *xc_index, float *prediction,
                int len_group) {
  int i = 0;

  for (i = 0; i < total; i++) {
    dets[i].bbox.x = prediction[xc_index[i] * len_group];
    dets[i].bbox.y = prediction[xc_index[i] * len_group + 1];
    dets[i].bbox.w = prediction[xc_index[i] * len_group + 2];
    dets[i].bbox.h = prediction[xc_index[i] * len_group + 3];
    dets[i].objectness = prediction[xc_index[i] * len_group + 4];
    dets[i].prob = prediction[xc_index[i] * len_group + 15];
  }

  return 0;
}

int nms_comparator_v3(const void *pa, const void *pb) {
  detection a = *(detection *)pa;
  detection b = *(detection *)pb;
  float diff = 0;
  diff = a.objectness - b.objectness;
  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0) return 0;
  float area = w * h;
  return area;
}

float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

float box_iou(box a, box b) {
  float I = box_intersection(a, b);
  float U = box_union(a, b);
  if (I == 0 || U == 0) {
    return 0;
  }
  return I / U;
}

int do_nms_sort(detection *dets, int total, float thresh) {
  int i = 0, j = 0;
  int total_result = total;

  qsort(dets, total, sizeof(detection), nms_comparator_v3);
  for (i = 0; i < total; ++i) {
    if (dets[i].prob == 0) continue;
    box a = dets[i].bbox;
    for (j = i + 1; j < total; ++j) {
      box b = dets[j].bbox;
      if (box_iou(a, b) > thresh) {
        dets[j].prob = 0;
        total_result--;
      }
    }
  }
  return total_result;
}

detr achieve_result(detr det_r, detection *dets, int num) {
  int i = 0, j = 0;
  for (i = 0; i < num; i++) {
    if (dets[i].prob == 0) continue;
    det_r.bbox[j].x = dets[i].bbox.x - (dets[i].bbox.w * 0.5);
    det_r.bbox[j].y = dets[i].bbox.y - (dets[i].bbox.h * 0.5);
    det_r.bbox[j].w = dets[i].bbox.w;
    det_r.bbox[j].h = dets[i].bbox.h;
    det_r.conf[j] = dets[i].objectness * dets[i].prob;
    j++;
  }

  return det_r;
}

detr non_max_suppression_face(float *prediction, int len_total, int len_group) {
  float conf_thres = 0.25;
  float iou_thres = 0.45;

  int *xc;
  int total_num = 0;
  xc = (int *)calloc(len_total, sizeof(int));
  total_num = create_xc(prediction, xc, conf_thres, len_total, len_group);

  int *xc_index = NULL;

  xc_index = create_xc_index(xc_index, total_num, xc, len_total);

  detection *dets;
  dets = (detection *)calloc(total_num, sizeof(detection));
  create_dets(dets, total_num, xc_index, prediction, len_group);

  int result_num = 0;
  result_num = do_nms_sort(dets, total_num, iou_thres);

  detr det_r;
  det_r.bbox = (box *)calloc(result_num, sizeof(box));
  det_r.conf = (float *)calloc(result_num, sizeof(float));
  det_r.num = result_num;
  det_r = achieve_result(det_r, dets, total_num);

  return det_r;
}

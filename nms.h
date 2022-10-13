#ifndef _NMS_H_
#define _NMS_H_

typedef struct image {
  int w;
  int h;
  int c;
  float *data;
} image;

typedef struct box {
  float x, y, w, h;
} box;

typedef struct detection {
  box bbox;
  float prob;
  float objectness;
} detection;

typedef struct detr {
  box *bbox;
  float *conf;
  int num;
} detr;

detr non_max_suppression_face(float *prediction, int len_total, int len_group);

#endif
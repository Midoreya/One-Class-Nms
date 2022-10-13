//---------------------//
#include <stdio.h>
#include <stdlib.h>
//---------------------//
#include "nms.h"

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

// gcc main.c nms.h nms.c

int main(int argc, char **argv) {
  int len_total = 24000;
  int len_group = 16;

  char *filepath = NULL;
  filepath = (char *)malloc(1024 * sizeof(char));
  filepath = argv[1];

  FILE *fp = NULL;
  fp = fopen(filepath, "r");
  float *tensor = NULL;
  tensor = (float *)malloc(len_group * len_total * sizeof(float));

  int i = 0;
  for (i = 0; i < len_group * len_total; i++) {
    fscanf(fp, "%f", &tensor[i]);
  }

  detr det;
  det = non_max_suppression_face(tensor, len_total, len_group);

  for (i = 0; i < det.num; i++) {
    printf("item[%d]: \n[%f, %f, %f, %f] %.2f%%,\n\n", i, det.bbox[i].x,
           det.bbox[i].y, det.bbox[i].w, det.bbox[i].h, det.conf[i] * 100);
  }

  return 0;
}
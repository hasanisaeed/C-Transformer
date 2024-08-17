#ifndef ENCODER_H
#define ENCODER_H

#include "../common/matrix_utils.h"
#include "../common/text_utils.h"

typedef struct
{
    double **output;
    int rows;
    int cols;
} EncoderOutput;

EncoderOutput *encoder(char *input);

#endif

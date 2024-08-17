#ifndef DECODER_H
#define DECODER_H

typedef struct
{
    double **output;
    int rows;
    int cols;
} DecoderOutput;

DecoderOutput *decoder(char *input, double **encoder_output, int encoder_rows, int encoder_cols);

#endif // DECODER_H
#ifndef PREDICT_WORD_UTILS_H
#define PREDICT_WORD_UTILS_H

double *flatten_matrix(double **matrix, int rows, int cols);
double *linear_layer(double *flattened_vector, int flattened_size, int vocab_size);
double *softmax_vector(double *logits, int size);
int select_max_probability(double *probs, int vocab_size);

#endif
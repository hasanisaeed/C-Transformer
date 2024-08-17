#ifndef TEXT_UTILS_H
#define TEXT_UTILS_H

#define EMBEDDING_VECTOR_DIM  4
#define WEIGHT_MATRIX_COLUMNS 4
#define INITIAL_CAPACITY      70
#define HEADS                 1
#define MAX_TEXT_LENGTH       10000

void process_text(char *text, char ***words, int *word_count, int *capacity);
double **create_embedding(char *prompt, int *rows, int *cols);
double **calculate_pe_vector(int rows, int cols);
double **concatenate_pe_and_embedding(double **pe, double **embedding, int rows, int cols);
double **add_and_normalize(double **matrix1, double **matrix2, int rows, int cols);
void read_text_file(const char *filename, char *text);
int length(char *str);

#endif // TEXT_UTILS_H
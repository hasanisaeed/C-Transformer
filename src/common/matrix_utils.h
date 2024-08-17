#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

double generate_random_number();
double **create_random_matrix(int rows, int cols);
double **create_2d_matrix(int rows, int cols);
double **create_pe_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void print_matrix(double **matrix, int rows, int cols, char *matrix_name);
double **add_matrices(double **A, double **B, int rows, int cols);
double **multiply_matrices(double **A, double **B, int rowsA, int colsA, int rowsB, int colsB);
double **transpose_matrix(double **A, int row, int col);
double **apply_function_on_matrix(double **matrix, int rows, int cols, double (*func)(double));
double **softmax(double **matrix, int rows, int cols);
double **copy_of(double **matrix, int rows, int cols);

#endif // MATRIX_UTILS_H

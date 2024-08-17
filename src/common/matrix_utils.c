#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_utils.h"

double generate_random_number()
{
    return (double)rand() / (double)RAND_MAX;
}

double **create_random_matrix(int row, int col)
{
    double **matrix = (double **)malloc(row * sizeof(double *));
    for (size_t i = 0; i < row; i++)
    {
        matrix[i] = (double *)malloc(col * sizeof(double));
    }

    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            matrix[i][j] = generate_random_number();
        }
    }
    return matrix;
}

double **create_2d_matrix(int row, int col)
{
    double **matrix = (double **)malloc(row * sizeof(double *));
    for (size_t i = 0; i < row; i++)
    {
        matrix[i] = (double *)malloc(col * sizeof(double));
    }

    return matrix;
}

double **create_pe_matrix(int row, int col)
{
    double **pe = create_2d_matrix(row, col);

    for (size_t i = 0; i < col; i++)
    {
        for (size_t j = 0; j < row; j++)
        {
            if (j % 2 == 0)
            {
                pe[i][j] = sin(i / pow(10000, 2 * i / (double)row));
            }
            else
            {
                pe[i][j] = cos(i / pow(10000, 2 * i / (double)row));
            }
        }
    }
    return pe;
}

void free_matrix(double **matrix, int row)
{
    for (size_t i = 0; i < row; i++)
    {

          free(matrix[i]);
    }
    free(matrix);
}

void print_matrix(double **matrix, int rows, int cols, char *matrix_name)
{
    if (matrix_name != NULL)
        printf("\n***** %s *****\n", matrix_name);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%f, ", matrix[i][j]);
        }
        printf("\n");
    }
}
double **add_matrices(double **A, double **B, int rows, int cols)
{
    double **temp = create_2d_matrix(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            temp[i][j] = A[i][j] + B[i][j];
        }
    }

    return temp;
}

double **multiply_matrices(double **A, double **B, int rowsA, int colsA, int rowsB, int colsB)
{
    double **temp = create_2d_matrix(rowsA, colsB);

    if (colsA != rowsB)
    {
        fprintf(stderr, "Error: Matrix multiplication not possible. colsA != rowsB\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            temp[i][j] = 0.0;
            for (int k = 0; k < colsA; k++)
            {
                temp[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return temp;
}

double **transpose_matrix(double **A, int row, int col)
{
    double **tmp = create_2d_matrix(col, row); 

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            tmp[j][i] = A[i][j];
        }
    }
    return tmp;
}

double **apply_function_on_matrix(double **matrix, int rows, int cols, double (*func)(double))
{
    double **tmp = create_2d_matrix(cols, rows);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            tmp[i][j] = func(matrix[i][j]);
        }
    }
    return tmp;
}

double **softmax(double **matrix, int rows, int cols)
{
    double **tmp = create_2d_matrix(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        double max_val = matrix[i][0];
        for (int j = 1; j < cols; j++)
        {
            if (matrix[i][j] > max_val)
            {
                max_val = matrix[i][j];
            }
        }

        double sum_exp = 0.0;
        for (int k = 0; k < cols; k++)
        {
            sum_exp += exp(matrix[i][k] - max_val);
        }

        for (int j = 0; j < cols; j++)
        {
            tmp[i][j] = exp(matrix[i][j] - max_val) / sum_exp;
        }
    }
    return tmp;
}

double **copy_of(double **matrix, int rows, int cols)
{
    double **tmp = create_2d_matrix(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            tmp[i][j] = matrix[i][j];
        }
    }

    return tmp;
}
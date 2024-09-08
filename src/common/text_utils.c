#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "text_utils.h"
#include "memory_utils.h"
#include "matrix_utils.h"

int is_word_unique(char *word, char **words, int word_count)
{
    for (int i = 0; i < word_count; i++)
    {
        if (strcmp(word, words[i]) == 0)
        {
            return 0; // Word is not unique
        }
    }
    return 1; // Word is unique
}

void process_text(char *text, char ***words, int *word_count, int *capacity)
{
    const char *delimiters = " .,!?)(-[]،/"; // Define delimiters to split the text
    char *word = strtok(text, delimiters);

    while (word != NULL)
    {
        if (is_word_unique(word, *words, *word_count))
        {
            (*words)[(*word_count)] = strdup(word);
            (*word_count)++;

            // Resize the array if necessary
            if (*word_count >= *capacity)
            {
                *capacity *= 2;
                *words = realloc(*words, (*capacity) * sizeof(char *));
            }
        }

        word = strtok(NULL, delimiters);
    }
}

double **create_embedding(char *prompt, int *rows, int *cols)
{
    *rows = EMBEDDING_VECTOR_DIM;
    *cols = length(prompt);

    double **embedding_vector = create_random_matrix(*rows, *cols);
    return embedding_vector;
}

double **calculate_pe_vector(int rows, int cols)
{
    double **pe = create_2d_matrix(rows, cols);

    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            double position = j / (double)rows;
            double div_term = pow(10000, 2 * i / (double)cols);
            if (div_term != 0)
            {
                double value = i / div_term;
                if (j % 2 == 0)
                {
                    pe[j][i] = sin(value);
                }
                else
                {
                    pe[j][i] = cos(value);
                }
            }
            else
            {
                pe[j][i] = 0; // handle div_term == 0
            }
        }
    }
    print_matrix(pe, rows, cols, "Positional Embedding");
    return pe;
}

double **concatenate_pe_and_embedding(double **pe, double **embedding, int rows, int cols)
{
    double **result = create_2d_matrix(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = pe[i][j] + embedding[i][j];
        }
    }

    print_matrix(result, rows, cols, "Concact PE & Embedding");

    return result;
}

int length(char *str)
{
    char str_cpy[strlen(str) + 1];
    strcpy(str_cpy, str);

    int word_count = 0;

    char *word = strtok(str_cpy, " ");
    while (word != NULL)
    {
        word_count++;
        word = strtok(NULL, " ");
    }

    return word_count;
}

void read_text_file(const char *filename, char *text)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    size_t length = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF && length < MAX_TEXT_LENGTH - 1)
    {
        text[length++] = ch;
    }
    text[length] = '\0';

    fclose(file);
}
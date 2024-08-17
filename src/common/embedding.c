#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "embedding.h"
#include "matrix_utils.h"

int _is_word_encoded(char *word, WordEncoding *encodings, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (strcmp(encodings[i].token, word) == 0)
        {
            return 1; // Word is already encoded
        }
    }
    return 0;
}

WordEncoding *create_encoding(char **words, int word_count)
{
    WordEncoding *encodings = (WordEncoding *)malloc(word_count * sizeof(WordEncoding));
    if (encodings == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int id = 1;
    int encoded_count = 0;

    printf("\n***** Encoding *****\n");

    for (int i = 0; i < word_count; i++)
    {
        if (!_is_word_encoded(words[i], encodings, encoded_count))
        {
            encodings[encoded_count].token = words[i];
            encodings[encoded_count].unique_id = id++;
            encoded_count++;

            printf("%s \t %d\n", words[i], id - 1);
        }
    }

    return encodings;
}

#include <stdlib.h>
#include "../decoder/decoder.h"
#include "../common/embedding.h"
#include "predict_word_utils.h"

char *predict_word(DecoderOutput *decoder_output, WordEncoding *words_encoding, int vocab_size, char **vocabulary)
{
    // Step 1: Flatten the decoder output
    double *flattened_vector = flatten_matrix(decoder_output->output, decoder_output->rows, decoder_output->cols);

    // Step 2: Pass through linear layer
    double *logits = linear_layer(flattened_vector, decoder_output->rows * decoder_output->cols, vocab_size);

    // Step 3: Apply softmax
    double *probs = softmax_vector(logits, vocab_size);

    // Step 4: Select the word with the highest probability
    int predicted_token_index = select_max_probability(probs, vocab_size);

    // Get the predicted word
    char *predicted_word = vocabulary[predicted_token_index];

    free(flattened_vector);
    free(logits);
    free(probs);

    return predicted_word;
}

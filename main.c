#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "src/common/text_utils.h"
#include "src/common/memory_utils.h"
#include "src/common/embedding.h"
#include "src/common/attention.h"
#include "src/common/matrix_utils.h"
#include "src/feed_forward/feed_forward_network.h"
#include "src/decoder/decoder.h"
#include "src/encoder/encoder.h"
#include "src/prediction/predict_word.h"

int main()
{
    srand(time(NULL));

    // Sample text
    char text[MAX_TEXT_LENGTH];

    read_text_file("my_text.txt", text);

    // Process text
    int vocab_size = 0, capacity = INITIAL_CAPACITY;
    char **words = allocate_memory(capacity);
    process_text(text, &words, &vocab_size, &capacity);

    // Create word encodings
    WordEncoding *words_encoding = create_encoding(words, vocab_size);

    // Embedding Vector
    char encoder_input[] = "Humanity thrives on empathy";

    EncoderOutput *encoder_output = encoder(encoder_input);

    print_matrix(encoder_output->output, encoder_output->rows, encoder_output->cols, "Encoder Output");

    char decoder_input[] = "<start> Love brings happiness <end>";

    DecoderOutput *decoder_output = decoder(decoder_input, encoder_output->output, encoder_output->rows, encoder_output->cols);

    print_matrix(decoder_output->output, decoder_output->rows, decoder_output->cols, "Decoder Output");

    char *predicted_word = predict_word(decoder_output, words_encoding, vocab_size, words);
    
    printf("\n>> Predicted Word: %s\n", predicted_word);

    free(decoder_output);
    free(encoder_output);
    free(words_encoding); // TODO
    free(words);

    printf("\n ***** Done! *****");
    return 0;
}
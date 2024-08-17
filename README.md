## C-Transformer

I created this repo to test my C programming skills and see how well I could handle it. I decided to implement a Transformer based on the famous [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

<p align="center">
    <img src=".assets/architecture.png" height = "500px" /><br/>
    <span>The Transformer - model architecture</span>
    
</p>

## Output

```
***** Encoding *****
Humanity         1
thrives          2
on               3
compassion       4
a                5
fundamental      6
trait            7
...

***** Positional Embedding *****
0.000000, 0.010000, 0.000200, 0.000003,
1.000000, 0.999950, 1.000000, 1.000000,
...

***** Single Head Attention *****
0.589237, 0.589154, 0.603617, 0.518480,
0.614376, 0.629398, 0.638245, 0.511109,
...

***** Feed Forward Output *****
0.260464, 1.185559, 0.141220, 1.783113,
...

***** Decoder Output *****
0.872295, -0.787095, -1.186268, 1.101068,
...

>> Predicted Word: true

```

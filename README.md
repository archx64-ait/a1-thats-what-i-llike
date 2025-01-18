# a1-thats-what-i-llike

Repository for NLP assignment 1

| Model            | Window Size | Training Loss | Training Time  | Syntactic Accuracy | Semantic Accuracy |
|------------------|-------------|---------------|----------------|--------------------|-------------------|
| Skipgram         | 2           | 9.083         | 13m 8s 511ms   | 0                  | 0                 |
| Skipgram (NEG)   | 2           | 7.27          | 15m 3s 810ms   | 0                  | 0                 |
| Glove            | 2           | 612.54        | 3m 3s 693ms    | 0                  | 0                 |
| Glove (Gensim)   | default     | -             | -              | 0                  | 0                 |

|Model           | Skipgram | NEG | GloVe | Glove (Gensim) | Y_true |
|----------------|----------|-----|-------|----------------|--------|
| MSE            | 26.602   | 29.366   | 28.357     | 26.602 | 5.12   |

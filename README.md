# a1-thats-what-i-llike

Repository for NLP assignment 1
student_id: st124974
student_name: Kaung Sithu

## How to run

There are 4 notebooks for training and saving the models, each with the corresponding name, and 2 notebooks for evaluating the models in the ```code``` folder.  ```evaluation.ipynb``` includes using the word analogies dataset to calculate syntactic and semantic accuracy and finding the correlation between the models' dot product and the provided similarity metrics. ```evaluation_gensim.ipynb``` contains the evaluation of the GloVe(Gensim) as I used different virtual environments for PyTorch and Gensim, they use different numpy versions.
The web application is in the ``app`` folder. Execute the following command to run the web application. The data for the nltk library is already included in nltk_data folder so that you don't have to download again.

Install the packages in ```req_gensim.txt``` to run ```glove_gensim.ipynb``` and ```evaluation_gensim.ipynb```

Install the packages in ```req_pytorch.txt``` to run the web application and the following notebooks.

- ```evaluation.ipynb```
- ```glove_scratch.ipynb```
- ```word2vev_skipgram.ipynb```
- ```word2vev_negative_sampling.ipynb```

```bash
python manage.py runserver
```

| Model            | Window Size | Training Loss | Training Time  | Syntactic Accuracy | Semantic Accuracy |
|------------------|-------------|---------------|----------------|--------------------|-------------------|
| Skipgram         | 2           | 9.083         | 13m 8s 511ms   | 0                  | 0                 |
| Skipgram (NEG)   | 2           | 7.27          | 15m 3s 810ms   | 0                  | 0                 |
| Glove            | 2           | 612.54        | 3m 3s 693ms    | 0                  | 0                 |
| Glove (Gensim)   | default     | -             | -              | 0                  | 0                 |

|Model           | Skipgram | NEG | GloVe | Glove (Gensim) | Y_true |
|----------------|----------|-----|-------|----------------|--------|
| MSE            | 26.602   | 29.366   | 28.357     | 26.602 | 5.12   |

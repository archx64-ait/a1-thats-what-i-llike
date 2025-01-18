import numpy as np
from scipy import spatial
from scipy.stats import spearmanr
from torch import nn
import torch
from nlp.vocab import vocab


class Skipgram(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, window_size, emb_size]
        all_embeds = self.embedding_u(all_vocabs)  # [batch_size, voc_size, emb_size]

        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, window_size]
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

        # [batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]
        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

        # scalar (loss must be scalar)
        nll = -torch.mean(
            torch.log(
                torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)
            )
        )

        return nll  # negative log likelihood


class SkipgramNegSampling(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_size)  # out embedding
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, window_size, emb_size]
        neg_embeds = -self.embedding_u(
            negative_words
        )  # [batch_size, window_size * num_neg, emb_size]

        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, window_size]

        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2))
        # [batch_size, k, emb_size] @ [batch_size, emb_size, 1] = [batch_size, k, 1]

        loss = self.logsigmoid(positive_score) + torch.sum(
            self.logsigmoid(negative_score), 1
        )

        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds


class GloVe(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, embed_size)  # out embedding

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        center_embeds = self.embedding_v(center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words
        )  # [batch_size, window_size, emb_size]

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

        # note that coocs already got log
        loss = weighting * torch.pow(
            inner_product + center_bias + target_bias - coocs, 2
        )  # [batch_size, window_size]

        return torch.sum(loss)  # scalar


def cos_sim(a, b):
    return 1 - spatial.distance.cosine(
        a, b
    )  # distance = 1 - similarlity, because scipy only gives distance


def load_specific_categories(file_path, semantic_category, syntactic_category):
    semantic = []
    syntactic = []
    current_group = None

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith(":"):
                if semantic_category in line.lower():
                    current_group = semantic
                elif syntactic_category in line.lower():
                    current_group = syntactic
                else:
                    current_group = None
            elif current_group is not None:
                words = line.strip().split()
                if len(words) == 4:
                    current_group.append(words)

    return semantic, syntactic


def find_closest_word(vec, embeddings, exclude_ids):
    max_similarity = -float("inf")
    best_idx = -1

    for idx, emb in enumerate(embeddings):
        if idx in exclude_ids:
            continue
        similarity = cos_sim(vec, emb)
        if similarity > max_similarity:
            max_similarity = similarity
            best_idx = idx

    return best_idx


def evaluate_analogies(analogy_data, word_to_idx, embeddings):
    correct = 0
    total = 0

    for word1, word2, word3, word4 in analogy_data:
        if all(word in word_to_idx for word in [word1, word2, word3, word4]):
            idx1 = word_to_idx[word1]
            idx2 = word_to_idx[word2]
            idx3 = word_to_idx[word3]
            idx4 = word_to_idx[word4]

            vec = embeddings[idx2] - embeddings[idx1] + embeddings[idx3]
            predicted_idx = find_closest_word(vec, embeddings, {idx1, idx2, idx3})
            if predicted_idx == idx4:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


def get_query_vector(query_word, model, word2index):
    # if query_word not in word2index:
    #     raise ValueError(f"Word '{query_word}' not in vocabulary.")

    word_idx = word2index.get(query_word)
    query_vector = model.embedding_v.weight[word_idx].data.cpu().numpy()
    return query_vector


def get_corpus_vectors(model):
    return model.embedding_v.weight.data.cpu().numpy()


def compute_top_k_dot_product(query_vector, corpus_vectors, k=10):
    # get dot product between the query and all corpus vectors
    dot_products = np.dot(corpus_vectors, query_vector)

    # get the top k indices and scores
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    top_k_scores = dot_products[top_k_indices]

    return top_k_indices.tolist(), top_k_scores.tolist()


def load_wordsim353(file_path):
    word_pairs = []
    human_scores = []

    with open(file_path, "r") as f:
        next(f)
        for line in f:
            word1, word2, score = line.strip().split()
            word_pairs.append((word1, word2))
            human_scores.append(float(score))

    return word_pairs, human_scores


def calculate_model_similarity(word_pairs, model, word_to_idx):
    model_scores = []
    embeddings = model.embedding_v.weight.data.cpu().numpy()

    for word1, word2 in word_pairs:
        if word1 in word_to_idx and word2 in word_to_idx:
            idx1 = word_to_idx[word1]
            idx2 = word_to_idx[word2]
            dot_product = np.dot(embeddings[idx1], embeddings[idx2])
            model_scores.append(dot_product)
        else:
            model_scores.append(None)  # Handle OOV (out-of-vocabulary) words
    return model_scores


def compute_spearman_correlation(human_scores, model_scores):
    valid_scores = [(h, m) for h, m in zip(human_scores, model_scores) if m is not None]
    filtered_human_scores, filtered_model_scores = zip(*valid_scores)

    correlation, _ = spearmanr(filtered_human_scores, filtered_model_scores)
    return correlation


def compute_mse(human_scores, model_scores):
    # remove null values from model_scores
    valid_scores = [(h, m) for h, m in zip(human_scores, model_scores) if m is not None]
    filtered_human_scores, filtered_model_scores = zip(*valid_scores)

    mse = np.mean(
        (np.array(filtered_model_scores) - np.array(filtered_human_scores)) ** 2
    )
    return mse


def compute_average_human_score(human_scores):
    return sum(human_scores) / len(human_scores)

word2index = {w: i for i, w in enumerate(vocab)}
index2word = {v:k for k, v in word2index.items()} 
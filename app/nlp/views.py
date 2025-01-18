from django.views.generic import TemplateView
from django.views.generic.edit import FormView
from nlp.forms import NLPForm
from typing import Any
from nlp.utils_scratch import *
from nlp.vocab import vocab
from django.urls import reverse_lazy
from django.shortcuts import redirect


class IndexView(TemplateView):
    template_name = "index.html"


class SuccessView(TemplateView):
    template_name = "success.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        result = self.request.GET.get("result")

        try:
            # Add the result to the context
            context["result"] = result

        except ValueError:
            context["result"] = [""]

        return context


class NLPFormView(FormView):
    voc_size = len(vocab)
    embedding_size = 2
    word2vec_skipgram = Skipgram(voc_size, embedding_size)
    word2vec_negativesampling = SkipgramNegSampling(voc_size, embedding_size)
    glove = GloVe(voc_size, embedding_size)

    form_class = NLPForm
    template_name = "nlp.html"

    def get_similar_words(self, model, query):
        similar_words = []
        query_vector = get_query_vector(query, model, word2index)
        corpus_vector = get_corpus_vectors(model)
        top_k_indices, _ = compute_top_k_dot_product(query_vector, corpus_vector, k=10)
        for idx in top_k_indices:
            similar_words.append(index2word[idx])
        
        beautified = ', '.join(similar_words)

        return beautified

    def predict(self, model, query):
        try:
            if model == "SG":
                return self.get_similar_words(self.word2vec_skipgram, query)

            elif model == "NG":
                return self.get_similar_words(self.word2vec_negativesampling, query)

            elif model == "GE":
                return self.get_similar_words(self.glove, query)
        except ValueError:
            return "The word you searched is not in vocabulary of the corpus"
            

    def form_valid(self, form):
        model = form.cleaned_data["model"]
        query = form.cleaned_data["query"]
        result = self.predict(model=model, query=query)
        return redirect(f"{reverse_lazy('nlp:success')}?result={result}")

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["results"] = getattr(self, "result", None)
        return context

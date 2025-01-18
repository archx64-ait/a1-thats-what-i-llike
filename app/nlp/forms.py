from django import forms

style = forms.Select(attrs={"class": "form-control"})


class NLPForm(forms.Form):
    model = forms.ChoiceField(
        choices=[
            ("SG", "Word2Vec(Skipgram)"),
            ("NG", "Word2Vec(Negativesampling)"),
            ("GE", "GloVe(Scratch)"),
            ("GN", "GloVe(Gensim)"),
        ],
        widget=style,
    )

    query = forms.CharField(required=True)

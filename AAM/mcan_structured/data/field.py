import os
import six
import nltk
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
import torch
from torchtext import data, vocab

class GensimKeyedVectorsAdapter:
    def __init__(self, kv_path: str):
        self.model = KeyedVectors.load(kv_path, mmap="r")
        self.dim = int(self.model.vector_size)
        self.supports_oov = False  # <— kv has no subword OOV

    def __contains__(self, token: str) -> bool:
        return token in self.model

    def __getitem__(self, token: str):
        import numpy as np
        import torch as _torch
        try:
            vec = self.model.get_vector(token)
        except KeyError:
            vec = np.zeros(self.dim, dtype=np.float32)  # kv: zero for OOV
        return _torch.tensor(vec, dtype=_torch.float32)

class GensimFastText:
    def __init__(self, model_path, **kwargs):
        self.model = load_facebook_vectors(model_path)
        self.dim = int(self.model.vector_size)
        self.supports_oov = True  # <— fastText can synthesize OOV via subwords

    def __contains__(self, token):
        # not strictly needed if you always use supports_oov path,
        # but keep it consistent:
        return token in self.model

    def __getitem__(self, token):
        import torch as _torch
        # use subword for OOV; returns np.array -> convert to float32 tensor
        try:
            vec = self.model.get_vector(token)  # works for IV and OOV
        except Exception:
            # extremely rare fallback
            import numpy as np
            vec = np.zeros(self.dim, dtype=np.float32)
        return _torch.tensor(vec, dtype=_torch.float32)

class MatchingVocab(vocab.Vocab):
    def extend_vectors(self, tokens, vectors):
        tot_dim = sum(v.dim for v in vectors)
        prev_len = len(self.itos)

        new_tokens = []
        for token in tokens:
            if token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
                new_tokens.append(token)
        self.vectors.resize_(len(self.itos), tot_dim)

        for i in range(prev_len, prev_len + len(new_tokens)):
            token = self.itos[i]
            assert token == new_tokens[i - prev_len]

            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert start_dim == tot_dim


class MatchingField(data.Field):
    vocab_cls = MatchingVocab
    _cached_vec_data = {}

    def __init__(self, tokenize='nltk', id=False, **kwargs):
        self.tokenizer_arg = tokenize
        self.is_id = id
        tokenize = MatchingField._get_tokenizer(tokenize)
        super(MatchingField, self).__init__(tokenize=tokenize, **kwargs)

    @staticmethod
    def _get_tokenizer(tokenizer):
        if tokenizer == 'nltk':
            return nltk.word_tokenize
        return tokenizer

    def preprocess_args(self):
        attrs = [
            'sequential', 'init_token', 'eos_token', 'unk_token', 'preprocessing',
            'lower', 'tokenizer_arg'
        ]
        args_dict = {attr: getattr(self, attr) for attr in attrs}
        for param, arg in list(six.iteritems(args_dict)):
            if six.callable(arg):
                del args_dict[param]
        return args_dict

    @classmethod
    def _get_vector_data(cls, vecs, cache):
        if not isinstance(vecs, list):
            vecs = [vecs]

        vec_datas = []
        for vec in vecs:
            print(vec)  # keep your debug
            if not isinstance(vec, vocab.Vectors):
                vec_name = vec
                vec_data = cls._cached_vec_data.get(vec_name)
                if vec_data is None:
                    # resolve absolute path if a cache dir is provided
                    model_path = os.path.join(cache, vec_name) if (cache is not None and not os.path.isabs(vec_name)) else vec_name
                    if vec_name.endswith('.bin'):
                        # Your existing FastText wrapper (already in your project)
                        vec_data = GensimFastText(model_path=model_path)
                    elif vec_name.endswith('.kv'):
                        # New: support KeyedVectors cache
                        vec_data = GensimKeyedVectorsAdapter(model_path)
                    else:
                        raise ValueError(f"Unknown vector format for {vec_name}")
                    cls._cached_vec_data[vec_name] = vec_data
                vec_datas.append(vec_data)
            else:
                vec_datas.append(vec)
        return vec_datas

    def build_vocab(self, *args, vectors=None, cache=None, **kwargs):
        if cache is not None:
            cache = os.path.expanduser(cache)
        if vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)

        super(MatchingField, self).build_vocab(*args, **kwargs)

        # Initialize vectors (concat of provided vector sets)
        if vectors:
            num_tokens = len(self.vocab.itos)
            total_dim = int(sum(vector.dim for vector in vectors))
            self.vocab.vectors = torch.zeros(num_tokens, total_dim)

            offset = 0
            for vector in vectors:
                dim = int(vector.dim)
                # If the vector source supports OOV (FastText), always assign;
                # otherwise (KeyedVectors), guard with membership.
                if getattr(vector, "supports_oov", False):
                    for token in self.vocab.itos:
                        self.vocab.vectors[self.vocab.stoi[token], offset:offset + dim] = vector[token]
                else:
                    for token in self.vocab.itos:
                        if token in vector:
                            self.vocab.vectors[self.vocab.stoi[token], offset:offset + dim] = vector[token]
                offset += dim

    def extend_vocab(self, *args, vectors=None, cache=None):
        sources = []
        for arg in args:
            if isinstance(arg, data.Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        tokens = set()
        for source in sources:
            for x in source:
                if not self.sequential:
                    tokens.add(x)
                else:
                    tokens.update(x)

        if self.vocab.vectors is not None and vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)
            self.vocab.extend_vectors(tokens, vectors)

    def numericalize(self, arr, *args, **kwargs):
        if not self.is_id:
            return super(MatchingField, self).numericalize(arr, *args, **kwargs)
        return arr


def reset_vector_cache():
    MatchingField._cached_vec_data = {}

import os

import nltk
import six
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import KeyedVectors
import torch
from torchtext import data, vocab

class GensimKeyedVectors:
    def __init__(self, model_path, binary=None):
        """
        If binary is None, load a .kv file with KeyedVectors.load().
        If binary is True/False, load word2vec format with load_word2vec_format().
        """
        if binary is None:
            self.model = KeyedVectors.load(model_path, mmap='r')
        else:
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
        self.dim = self.model.vector_size

    def __getitem__(self, token):
        try:
            return torch.tensor(self.model[token])
        except KeyError:
            return torch.zeros(self.dim)
class GensimFastText:
    def __init__(self, model_path, **kwargs):
        # Load the FastText binary file using gensim
        self.model = load_facebook_vectors(model_path)
        self.dim = self.model.vector_size

    def __getitem__(self, token):
        # Generate vector for OOV words using subword information
        if token in self.model:
            return torch.tensor(self.model[token])
        else:
            # Use subword information to create vectors for OOV tokens
            return torch.tensor(self.model.get_vector(token))



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
            print(vec)
            if not isinstance(vec, vocab.Vectors):
                vec_name = vec
                vec_data = cls._cached_vec_data.get(vec_name)
                if vec_data is None:
                      if vec_name.endswith('.bin'):  # For Gensim FastText binary format
                          model_path = os.path.join(cache, vec_name)
                          vec_data = GensimFastText(model_path=model_path)
                      elif vec_name.endswith('.kv'):  # Gensim KeyedVectors (.kv)
                          model_path = os.path.join(cache, vec_name)
                          vec_data = GensimKeyedVectors(model_path=model_path, binary=None)
                      elif vec_name.endswith('.vec') or vec_name.endswith('.txt'):
                          # word2vec text format
                          model_path = os.path.join(cache, vec_name)
                          vec_data = GensimKeyedVectors(model_path=model_path, binary=False)
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

        # Initialize vectors as zero tensors if not already done
        if vectors:
            num_tokens = len(self.vocab.itos)
            total_dim = sum(vector.dim for vector in vectors)
            self.vocab.vectors = torch.zeros(num_tokens, total_dim)

            # Assign vectors to tokens in the vocabulary
            for vector in vectors:
                for token in self.vocab.itos:
                    if token in vector.model:
                        self.vocab.vectors[self.vocab.stoi[token]] += vector[token].clone().detach()

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

        if self.vocab.vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)
            self.vocab.extend_vectors(tokens, vectors)

    def numericalize(self, arr, *args, **kwargs):
        if not self.is_id:
            return super(MatchingField, self).numericalize(arr, *args, **kwargs)
        return arr


def reset_vector_cache():
    MatchingField._cached_vec_data = {}

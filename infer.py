from nltk.tokenize import TreebankWordTokenizer
import torch
import torch.nn.functional as F

# from hydra import compose, initialize
# from omegaconf import OmegaConf

import hydra

from dataloader import get_label_maps, load_word_map
from models import TransformerClassifier
from store import Checkpointer
from utils import get_device


class SentenceProcessor:

    def __init__(self, wordmap, max_len):
        self.word_tokenizer = TreebankWordTokenizer()
        self.wordmap = wordmap;
        self.max_len = max_len

    def process(self, sentence: str):
        sentence = self.word_tokenizer.tokenize(sentence)[:self.max_len]

        encoded_sent = list(
            map(lambda w: self.wordmap.get(w, self.wordmap['<unk>']), sentence)
        ) + [0] * (self.max_len - len(sentence))

        encoded_sent = torch.LongTensor(encoded_sent).unsqueeze(0)

        return encoded_sent


def infer(sentence, model, word_map, max_len, label_map, rev_label_map):
    sp = SentenceProcessor(word_map, max_len)

    encoded_sent = sp.process(sentence)
    encoded_sent = encoded_sent.to(get_device())
    # run through model
    scores = model(encoded_sent)

    scores = scores.squeeze(0)  # (n_classes)
    scores = F.softmax(scores, dim=0)

    score, prediction = scores.max(dim=0)

    prediction = 'Category: {category}, Probability: {score:.2f}%'.format(
        category=rev_label_map[prediction.item()],
        score=score.item() * 100
    )
    return prediction


@hydra.main(config_path="config/conf", config_name="config")
def run(cfg):
    word_map = load_word_map(cfg.paths.data)
    q_limit = cfg.params.query_limit

    checkpoint = Checkpointer.load_checkpoint(cfg.checkpoint.checkpoint_id, cfg.checkpoint.path,
                                              str(get_device()))

    label_map, rev_label_map = get_label_maps()
    vocab_size = len(word_map)
    n_classes = len(label_map)

    model_kwargs = {
        'dim': cfg.params.emb_size,
        'q_len': cfg.params.query_limit,
        'ffn_hidden_layer_size': cfg.params.ffn_hidden_size,
        'n_heads': cfg.params.n_heads,
        'n_encoder': cfg.params.n_encoders,
        'classfifier_hidden_layer_size': cfg.params.classifier_hidden_layer_size,
        'dropout': cfg.params.dropout
    }

    model = TransformerClassifier(n_classes, vocab_size, None, **model_kwargs)
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(get_device())
    model.eval()
    #sentence = "French president says Croissant is not really french."
    sentence = "FC Barcelona finally accepts it has turned into shit."

    prediction = infer(sentence, model, word_map, q_limit, label_map, rev_label_map)
    print(sentence)
    print(prediction)


if __name__ == "__main__":
    # initialize(config_path="config/conf", job_name="infer")
    # cfg = compose(config_name="config", overrides=[])
    # cfg = OmegaConf.create(OmegaConf.to_container(cfg))

    run()

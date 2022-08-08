# Copied from: https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/unixcoder.py

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from state import State
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import os
from pathlib import Path


class UniXcoder(nn.Module):
    def __init__(self, model_name, state_dict=None):
        """
            Build UniXcoder.
            Parameters:
            * `model_name`- huggingface model card name. e.g. microsoft/unixcoder-base
        """
        super(UniXcoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name, use_fast=True)
        self.config = RobertaConfig.from_pretrained(model_name)
        self.config.is_decoder = True
        self.model = RobertaModel.from_pretrained(
            model_name, config=self.config)

        if state_dict is not None:
            self.model.load_state_dict(torch.load(state_dict))

        self.register_buffer("bias", torch.tril(torch.ones(
            (1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024))
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.tokenizer.add_tokens(["<mask0>"], special_tokens=True)
        #self.tokenizer.add_tokens(["<END>"], special_tokens=True)

    def tokenize(self, inputs, mode="<encoder-only>", max_length=512, padding=False):
        """ 
        Convert string to token ids 

        Parameters:
        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length. 
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        assert mode in ["<encoder-only>",
                        "<decoder-only>", "<encoder-decoder>"]

        tokenizer = self.tokenizer

        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[:max_length-4]
                tokens = [tokenizer.cls_token, mode,
                          tokenizer.sep_token] + tokens + [tokenizer.sep_token]
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length-3):]
                tokens = [tokenizer.cls_token, mode,
                          tokenizer.sep_token] + tokens
            else:
                tokens = tokens[:max_length-5]
                tokens = [tokenizer.cls_token, mode,
                          tokenizer.sep_token] + tokens + [tokenizer.sep_token]

            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            tokens_ids.append(tokens_id)

        if padding:
            cur_max_length = len(max(tokens_ids, key=len))
            tokens_ids = list(map(
                lambda l: l + [self.config.pad_token_id] * (cur_max_length-len(l)), tokens_ids))
        return tokens_ids

    def decode(self, source_ids):
        """ Convert token ids to string """
        predictions = []
        for x in source_ids:
            prediction = []
            for y in x:
                t = y.cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = self.tokenizer.decode(
                    t, clean_up_tokenization_spaces=False)
                prediction.append(text)
            predictions.append(prediction)
        return predictions

    def forward(self, source_ids):
        """ Obtain token embeddings and sentence embeddings """
        mask = source_ids.ne(self.config.pad_token_id)
        token_embeddings = self.model(
            source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        sentence_embeddings = (
            token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return token_embeddings, sentence_embeddings

    def generate(self, source_ids, decoder_only=True, eos_id=None, beam_size=5, max_length=64):
        """ Generate sequence given context (source_ids) """

        # Set encoder mask attention matrix: bidirectional for <encoder-decoder>, unirectional for <decoder-only>
        if decoder_only:
            mask = self.bias[:, :source_ids.size(-1), :source_ids.size(-1)]
        else:
            mask = source_ids.ne(self.config.pad_token_id)
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        if eos_id is None:
            eos_id = self.config.eos_token_id

        device = source_ids.device

        # Decoding using beam search
        preds = []
        zero = torch.LongTensor(1).fill_(0).to(device)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        length = source_ids.size(-1)
        encoder_output = self.model(source_ids, attention_mask=mask)
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1, :, :source_len[i]].repeat(beam_size, 1, 1, 1) for x in y]
                       for y in encoder_output.past_key_values]
            beam = Beam(beam_size, eos_id, device)
            input_ids = beam.getCurrentState().clone()
            context_ids = source_ids[i:i+1,
                                     :source_len[i]].repeat(beam_size, 1)
            out = encoder_output.last_hidden_state[i:i +
                                                   1, :source_len[i]].repeat(beam_size, 1, 1)
            for _ in range(max_length):
                if beam.done():
                    break
                if _ == 0:
                    hidden_states = out[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = beam.getCurrentState().clone()
                else:
                    length = context_ids.size(-1)+input_ids.size(-1)
                    out = self.model(input_ids, attention_mask=self.bias[:, context_ids.size(-1):length, :length],
                                     past_key_values=context).last_hidden_state
                    hidden_states = out[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState().clone()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]
                              * (max_length-len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds


class Beam(object):
    def __init__(self, size, eos, device):
        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(0).to(device)]
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.nextYs[-1].view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


# Partially coied from: https://github.com/microsoft/CodeBERT/blob/567dd49a4b916835f93fb95709de714b8772fea2/UniXcoder/downstream-tasks/code-search/model.py

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, inputs):
        outputs = self.encoder(inputs)[1]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)

    def save(self, suffix):
        output_dir = Path(".")
        model_to_save = self.encoder.model
        output_dir = os.path.join(output_dir, 'model-{}.bin'.format(suffix))
        torch.save(model_to_save.state_dict(), output_dir)
        print("Saved model to {}".format(output_dir))

    def get_texts_tokens(self, texts, state):
        tokens = self.encoder.tokenize(
            texts, max_length=512, mode="<encoder-only>", padding=True)
        return torch.tensor(tokens).to(state.device)


def reload_model(state: State, state_dict):
    unixcoder_model = UniXcoder(
        model_name=state.config.unixcoder_model_path, state_dict=state_dict)
    unixcoder_model.to(state.device)
    return unixcoder_model


def get_text_tokens(state: State, model, text):
    tokens = model.tokenize(
        [text], max_length=512, mode="<encoder-only>")
    return torch.tensor(tokens).to(state.device)


def get_text_embedding(state: State, model, text):
    source_ids = get_text_tokens(state, model, text)
    _, embeddings = model(source_ids)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()[0]


@torch.no_grad()
def get_nb_embeddings(state: State, model, nb):
    res = {}

    batch_size = state.config.batch_size
    n_chunks = len(nb) / min(len(nb), batch_size)

    nb = nb.sort_values(by="source", key=lambda x: x.str.len())
    for nb in np.array_split(nb, n_chunks):
        texts = nb['source'].to_numpy()

        tokens = model.tokenize(texts, max_length=512,
                                mode="<encoder-only>", padding=True)
        source_ids = torch.tensor(tokens).to(state.device)
        _, embeddings = model(source_ids)
        normalized = torch.nn.functional.normalize(
            embeddings, p=2, dim=1).cpu()

        for key, val in zip(nb['source'].index, normalized):
            res[key] = val

    res['END'] = get_text_embedding(state, model, 'END')

    return res


class EnsembleModel(nn.Module):
    def __init__(self, state, state_dict=None):
        super(EnsembleModel, self).__init__()
        self.encoder = reload_model(state, state_dict=None)
        self.top = nn.Linear(768 + 6, 2)
        self.softmax = nn.Softmax(dim=1)
        self.name = ""
        if state_dict is not None:
            self.name = state_dict
            self.load_state_dict(torch.load(
                state_dict, map_location=state.device))
        self.to(state.device)

    def forward(self, inputs, additional_features, device):
        outputs = self.encoder(inputs)[1]
        joined = torch.cat((outputs, additional_features), 1).to(device)
        per_model = self.top(joined)
        return self.softmax(per_model)

    def save(self, suffix):
        output_dir = Path(".")
        output_path = os.path.join(
            output_dir, 'ensemble-model-{}.bin'.format(suffix))
        torch.save(self.state_dict(), output_path)
        print("Saved model to {}".format(output_path))

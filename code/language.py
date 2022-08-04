from typing import Sequence
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from common import get_markdown_cells


def detect_lang(x):
    try:
        return detect(x)
    except:
        return 'other'


def most_frequent(List):
    return max(set(List), key=List.count)


def detect_nb_lang(nb):
    markdowns = get_markdown_cells(nb)
    langs = []
    for cell_id in markdowns:
        langs.append(detect_lang(nb.loc[cell_id]['source']))
    langs.sort()
    best = most_frequent(langs)
    if best == 'en':
        without_en = list(filter(lambda a: a != 'en', langs))
        if len(without_en) != 0:
            best2 = most_frequent(without_en)
            cnt = langs.count(best2)
            if cnt > 2 and cnt * 3 >= len(langs):
                return best2
    return best


class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        # TODO: max_len?
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]


translators = {}


def get_translator(source: str, dest: str):
    key = source + "->" + dest
    if key not in translators:
        translators[key] = Translator(source, dest)
    return translators[key]

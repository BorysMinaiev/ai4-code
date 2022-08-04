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
    return most_frequent(langs)

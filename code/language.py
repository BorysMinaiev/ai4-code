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

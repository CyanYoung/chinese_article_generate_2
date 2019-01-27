import json

import re


path_poetry = 'dict/poetry.json'
with open(path_poetry, 'r') as f:
    poetry = json.load(f)


def retrieve():
    poet = input('poet: ')
    if poet in poetry:
        key = input('title: ')
        titles, texts = list(), list()
        for cand in poetry[poet].keys():
            if re.findall(key, cand):
                titles.append(cand)
                texts.extend(poetry[poet][cand])
        if titles:
            for title, text in zip(titles, texts):
                print('%s：%s' % (title, text))
        else:
            print('no title')
    else:
        print('no poet')


if __name__ == '__main__':
    while True:
        retrieve()

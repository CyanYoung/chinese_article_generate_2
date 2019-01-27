import pandas as pd


def flat_read(path, field):
    nest_items = pd.read_csv(path, usecols=[field], keep_default_na=False).values
    items = list()
    for nest_item in nest_items:
        items.append(nest_item[0])
    return items


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError

import json

from random import shuffle


def save(path, quaples):
    with open(path, 'w') as f:
        f.write('poet,title,text' + '\n')
        for num, title, poet, text in quaples:
            f.write(poet + ',' + title + ',' + text + '\n')


def check(quaples):
    nums = [fields[0] for fields in quaples]
    for i in range(len(nums) - 1):
        num1, num2 = [int(num) for num in nums[i].split('_')]
        next_num1, next_num2 = [int(num) for num in nums[i + 1].split('_')]
        if not (num1 == next_num1 and next_num2 - num2 == 1) and not \
               (next_num1 - num1 == 1 and next_num2 == 1):
            print('{}_{} -> {}_{}'.format(num1, num2, next_num1, next_num2))


def prepare(path_univ, path_train, path_dev, path_test, path_poetry, detail):
    quaples = list()
    poetry = dict()
    with open(path_univ, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 4:
                continue
            quaples.append(fields)
            num, title, poet, text = fields
            if poet not in poetry:
                poetry[poet] = dict()
            if title not in poetry[poet]:
                poetry[poet][title] = list()
            poetry[poet][title].append(text)
    if detail:
        check(quaples)
    shuffle(quaples)
    bound1 = int(len(quaples) * 0.7)
    bound2 = int(len(quaples) * 0.9)
    save(path_train, quaples[:bound1])
    save(path_dev, quaples[bound1:bound2])
    save(path_test, quaples[bound2:])
    with open(path_poetry, 'w') as f:
        json.dump(poetry, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path_univ = 'data/univ.txt'
    path_train = 'data/train.csv'
    path_dev = 'data/dev.csv'
    path_test = 'data/test.csv'
    path_poetry = 'dict/poetry.json'
    prepare(path_univ, path_train, path_dev, path_test, path_poetry, detail=False)

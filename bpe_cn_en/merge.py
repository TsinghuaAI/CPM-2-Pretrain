s = set()

with open('vocab.txt') as fin:
    for line in fin:
        line = line.strip()
        s.add(line)

with open('vocab-en.txt') as fin:
    with open('vocab-en-new.txt', 'w') as fout:
        for line in fin:
            line = line.strip()
            if line not in s:
                fout.write('{}\n'.format(line))


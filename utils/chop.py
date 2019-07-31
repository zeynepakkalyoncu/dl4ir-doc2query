from nltk.tokenize import sent_tokenize


def chop_by_token(path, window=20, stride=10):
    with open(path, 'r') as in_file, open('{}_t_w{}_s{}.txt'.format(path[:path.find('.txt')], window, stride), 'w') as out_file:
        for in_line in in_file:
            start = 0
            tokens = in_line.strip().split(' ')
            while start + window <= len(tokens):
                out_file.write('{}\n'.format(' '.join(tokens[start:(start + window)])))
                start += stride
            out_file.write('{}\n'.format(' '.join(tokens[start:])))


def chop_by_sent(path, window=3, stride=2):
    with open(path, 'r') as in_file, open('{}_s_w{}_s{}.txt'.format(path[:path.find('.txt')], window, stride), 'w') as out_file:
        for in_line in in_file:
            start = 0
            sents = sent_tokenize(in_line)

            while start + window <= len(sents):
                out_file.write('{}\n'.format(' '.join(sents[start:(start + window)])))
                start += stride


if __name__ == '__main__':
    chop_by_token('../msmarco_data/opennmt_format/src-collection.txt')

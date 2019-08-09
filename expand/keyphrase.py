import pke

import time
import sys
import warnings
warnings.filterwarnings("ignore")


index = sys.argv[1]

# initialize keyphrase extraction model
extractor = pke.unsupervised.YAKE()

start = time.time()

with open('msmarco-docs.{}'.format(index), 'r') as in_file, open('msmarco-docs-yake.{}'.format(index), 'w') as out_file:
    for line in in_file:
        docid, url, title, doc = line.strip().split('\t')
        with open('tmp.txt', 'w') as tmp_file:
            tmp_file.write(title + ' ' + doc)
        extractor.load_document(input='tmp.txt', encoding='utf-8', language='en')

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives (i.e. `(Noun|Adj)*`)
        extractor.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        extractor.candidate_weighting()

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        keyphrases = extractor.get_n_best(n=3)

        expansion = ' '.join([kp[0] for kp in keyphrases]).strip()
        out_file.write('{}\n'.format(expansion))

print(time.time() - start)

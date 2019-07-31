import pke

import sys
import warnings
warnings.filterwarnings("ignore")

# initialize keyphrase extraction model
extractor = pke.unsupervised.YAKE()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy

for i in range(0, 8841824):
    extractor.load_document(input='lines/passage.{}'.format(i), encoding='utf-8', language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=3)

    # print(keyphrases)

    expansion = ' '.join([kp[0] for kp in keyphrases]).strip()
    out_file.write('{}\n'.format(expansion))


# with open(sys.argv[1], 'r') as in_file, open(sys.argv[2], 'w') as out_file:
#     for line in in_file:
#         with open(sys.argv[3], 'w', encoding='utf-8') as tmp_file:
#             tmp_file.write(line)
#         with open(sys.argv[3], 'r+', encoding='utf-8') as tmp_file:
#             extractor.load_document(input=sys.argv[3], encoding='utf-8', language='en')
#
#             # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
#             # and adjectives (i.e. `(Noun|Adj)*`)
#             extractor.candidate_selection()
#
#             # candidate weighting, in the case of TopicRank: using a random walk algorithm
#             extractor.candidate_weighting()
#
#             # N-best selection, keyphrases contains the 10 highest scored candidates as
#             # (keyphrase, score) tuples
#             keyphrases = extractor.get_n_best(n=3)
#
#             # print(keyphrases)
#
#             expansion = ' '.join([kp[0] for kp in keyphrases]).strip()
#             out_file.write('{}\n'.format(expansion))

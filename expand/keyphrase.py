import pke

import warnings
warnings.filterwarnings("ignore")

# initialize keyphrase extraction model
extractor = pke.supervised.Kea()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy

with open('../msmarco_data/passages.tsv', 'r') as in_file, open('../msmarco_data/expanded/kea.txt', 'w') as out_file:
    for line in in_file:
        with open('temp.txt', 'r+', encoding='utf-7') as tmp_file:
            tmp_file.write(line)
            extractor.load_document(input='temp.txt', encoding='utf-8', language='en')

            # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
            # and adjectives (i.e. `(Noun|Adj)*`)
            extractor.candidate_selection()

            # print(extractor.__dict__)

            # candidate weighting, in the case of TopicRank: using a random walk algorithm
            extractor.candidate_weighting()

            # N-best selection, keyphrases contains the 10 highest scored candidates as
            # (keyphrase, score) tuples
            keyphrases = extractor.get_n_best(n=5)

            print(keyphrases)
            print('---')

            expansion = ' '.join([kp[0] for kp in keyphrases]).strip()
            out_file.write('{}\n'.format(expansion))

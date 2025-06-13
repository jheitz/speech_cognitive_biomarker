import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import random
import os
from collections import defaultdict
from functools import reduce

from util.helpers import safe_divide
from data_analysis.util.decorators import cache_to_file_decorator

class LinguisticFeatureLiteratureCalculator:

    def __init__(self, doc, constants=None):
        self.CONSTANTS = constants
        self.doc = doc
        self.words_enriched = [word for sent in doc.sentences for word in sent.words]
        self.words_raw = [word.text for sent in doc.sentences for word in sent.words if word.pos != 'PUNCT']
        self.sentences_enriched = doc.sentences
        self.sentences_raw = [[w.text for w in sentence.words if w.pos != 'PUNCT'] for sentence in doc.sentences]


    def n_words(self):
        """ Number of words """
        return len(self.words_raw)

    def n_unique(self):
        """ Number of unique words / types """
        return len(set(self.words_raw))

    def word_length(self):
        """ Average length of words in letters """
        return safe_divide(sum([len(word) for word in self.words_raw]), self.n_words())

    def sentence_length(self):
        """ Average length of sentences in words """
        return safe_divide(sum([len(sentence) for sentence in self.sentences_raw]), len(self.sentences_raw))

    def pos_counts(self):
        """
        Count number of POS tags in doc
        """
        # universal POS (UPOS) tags (https://universaldependencies.org/u/pos/)
        count_pos = defaultdict(lambda: 0)
        # treebank-specific POS (XPOS) tags (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
        count_xpos = defaultdict(lambda: 0)
        for word in self.words_enriched:
            count_pos[word.pos] += 1
            count_xpos[word.xpos] += 1

        return count_pos, count_xpos

    def constituency_rules_count(self):
        """
        Returns counts of all CFG production rules from the constituency parsing of the doc
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def constituency_rules(doc):
            def get_constituency_rule_recursive(tree):
                label = tree.label
                children = tree.children
                if len(children) == 0:  # this is a leaf
                    return []

                # find children label, if child is not a leaf. also ignore full stops
                children_label = "_".join([c.label for c in children if len(c.children) > 0 and c.label != '.'])
                if len(children_label) > 0:
                    rule = f"{label} -> {children_label}"
                    # print(f"{rule}     ({tree})")
                else:
                    return []

                children_rules = [get_constituency_rule_recursive(c) for c in children]
                children_rules_flat = [ll for l in children_rules for ll in l]

                return children_rules_flat + [rule]

            all_rules = []
            for sentence in doc.sentences:
                tree = sentence.constituency
                rules = get_constituency_rule_recursive(tree)
                all_rules += rules

            return all_rules

        cfg_rules = constituency_rules(self.doc)
        count_rules = defaultdict(lambda: 0)
        for rule in cfg_rules:
            count_rules[rule] += 1

        return count_rules

    def get_constituents(self):
        """
        Get constituents of the document, with corresponding text (concatenation of the leaf words)
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def get_leafs_text(tree):
            if len(tree.children) == 0:
                return tree.label
            return " ".join([get_leafs_text(c) for c in tree.children])

        def get_constituents_recursive(tree):
            label = tree.label
            if len(tree.children) == 0:
                return []

            children_constituents = [get_constituents_recursive(c) for c in tree.children]
            children_constituents_flat = [ll for l in children_constituents for ll in l]

            text = get_leafs_text(tree)
            return children_constituents_flat + [{'label': label, 'text': text}]

        all_constituents = []
        for sentence in self.sentences_enriched:
            tree = sentence.constituency
            rules = get_constituents_recursive(tree)
            all_constituents += rules
        return all_constituents

    def mattr(self, window_length=20):
        """
        Moving-Average Type-Token Ratio (MATTR)
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        if len(self.words_raw) == 0:
            return 0
        elif len(self.words_raw) < (window_length + 1):
            ma_ttr = len(set(self.words_raw)) / len(self.words_raw)
        else:
            sum_ttr = 0
            for x in range(len(self.words_raw) - window_length + 1):
                window = self.words_raw[x:(x + window_length)]
                sum_ttr += len(set(window)) / float(window_length)
            ma_ttr = sum_ttr / len(self.words_raw)
        return ma_ttr

    def ndw50(self, window_length=50):
        """
        Number of different words in a 50-word sample (mean over multiple runs)
        Inspired from Predicting Montreal Cognitive Assessment Scores From Measures of Speech and Language (Wisler, Fletcher, McAuliffe, 2020)
        """
        if len(self.words_raw) == 0:
            return 0
        elif len(self.words_raw) < (window_length + 1):
            ndw = len(set(self.words_raw)) / len(self.words_raw) * 50
        else:
            ndw_sum = 0
            n_samples = 100
            for x in range(n_samples):
                sample = random.sample(self.words_raw, 50)
                ndw_sum += len(set(sample))
            ndw = ndw_sum / n_samples
        return ndw

    def ttr(self):
        """
        Type-token ratio
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        ntokens = len(self.words_raw)
        ntypes = len(set(self.words_raw))
        return safe_divide(ntypes, ntokens)

    def brunets_index(self):
        """
        Brunét's index, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_tokens = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_tokens > 0:
            return n_tokens ** n_types ** (-0.165)
        else:
            return 0

    def honores_statistic(self):
        """
        Honoré's statistic, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_words_with_one_occurence = len(list(filter(lambda w: self.words_raw.count(w) == 1, self.words_raw)))
        n_words = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_words == 0:
            return 0
        elif (1 - n_words_with_one_occurence / n_types) == 0:
            return 0
        else:
            return (100 * np.log(n_words)) / (1 - n_words_with_one_occurence / n_types)


    def _count_syllables(self, word):
        """
        Simple syllable counting, from on  https://github.com/cdimascio/py-readability-metrics/blob/master/readability/text/syllables.py
        """
        word = word if type(word) is str else str(word)
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub('(?:[^laeiouy]es|[^laeiouy]e)$', '', word)  # removed ed|
        word = re.sub('^y', '', word)
        matches = re.findall('[aeiouy]{1,2}', word)
        return len(matches)

    def flesch_kincaid(self):
        """
        Flesch–Kincaid grade level
        Based on https://github.com/cdimascio/py-readability-metrics/blob/master/readability/scorers/flesch_kincaid.py
        Formula adapted according to https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
        """
        words = [word for sent in self.sentences_raw for word in sent]

        syllables_per_word = [self._count_syllables(w) for w in words]
        avg_syllables_per_word = safe_divide(sum(syllables_per_word), len(words))

        avg_words_per_sentence = safe_divide(len(words), len(self.sentences_raw))

        return (0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word) - 15.59

    #@cache_to_file_decorator()
    def _cos_dist_between_sentences(self, sentences_raw):
        """
        Average cosine distance between utterances, and proportion of sentence pairs whose cosine distance is less than or equal to 0.5
        Based on and adapted from https://github.com/vmasrani/dementia_classifier/blob/1f48dc89da968a6c9a4545e27b162c603eb9a310/dementia_classifier/feature_extraction/feature_sets/psycholinguistic.py#L686
        """
        stop = nltk.corpus.stopwords.words('english')
        stemmer = nltk.PorterStemmer()
        def not_only_stopwords(text):
            unstopped = [w for w in text.lower() if w not in stop]
            return len(unstopped) != 0
        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]
        def normalize(text):
            text = str(text).lower()
            return stem_tokens(nltk.word_tokenize(text))

        def cosine_sim(text1, text2):
            # input: list of raw utterances
            # returns: list of cosine similarity between all pairs
            if not_only_stopwords(text1) and not_only_stopwords(text2):
                # Tfid raises error if text contain only stopwords. Their stopword set is different
                # than ours so add try/catch block for strange cases
                try:
                    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')  # Punctuation remover
                    tfidf = vectorizer.fit_transform([text1, text2])
                    return ((tfidf * tfidf.T).A)[0, 1]
                except ValueError as e:
                    print("Error:", e)
                    print('Returning 0 for cos_sim between: "', text1, '" and: "', text2, '"')
                    return 0
            else:
                return 0

        def compare_all_utterances(uttrs):
            # input: list of raw utterances
            # returns: (float)average similarity over all similarities
            similarities = []
            for i in range(len(uttrs)):
                for j in range(i + 1, len(uttrs)):
                    similarities.append(cosine_sim(uttrs[i], uttrs[j]))
            return similarities

        def avg_cos_dist(similarities):
            # returns:(float) Minimum similarity over all similarities
            return safe_divide(reduce(lambda x, y: x + y, similarities), len(similarities))

        def proportion_below_threshold(similarities, thresh):
            # returns: proportion of sentence pairs whose cosine distance is less than or equal to a threshold
            valid = [s for s in similarities if s <= thresh]
            return safe_divide(len(valid), float(len(similarities)))

        if len(sentences_raw) < 2:
            # only one sentence -> return 0
            return 0, 0
        sentences_as_text = [" ".join(sentence) for sentence in sentences_raw]
        similarities = compare_all_utterances(sentences_as_text)
        return avg_cos_dist(similarities), proportion_below_threshold(similarities, 0.5)

    def cos_dist_between_sentences(self):
        return self._cos_dist_between_sentences(self.sentences_raw)


    @cache_to_file_decorator(verbose=False)
    def get_english_dictionary(self):
        # word list from spell checker library
        # https://github.com/barrust/pyspellchecker/blob/master/spellchecker/resources/en.json.gz
        with open(os.path.join(self.CONSTANTS.RESOURCES_DIR, 'english_dictionary.csv')) as f:
            words = f.readlines()
            words = [w.strip() for w in words]
            # add some words not in dictionary but okay, such as n't token from "don't"
            words += ["n't", "'re", "'ll"]
            return words

    def not_in_dictionary(self):
        words_greater_than_two = [w for w in self.words_raw if len(w) > 2]  # only consider words greater than length 2
        #not_found = [w for w in words_greater_than_two if w.lower() not in nltk.corpus.words.words()]
        not_found = [w for w in words_greater_than_two if w.lower() not in self.get_english_dictionary()]
        print("Words not in dictionary: ", not_found)
        return safe_divide(len(not_found), len(words_greater_than_two))

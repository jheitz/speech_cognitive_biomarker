import os
import stanza
from collections import defaultdict
import pandas as pd
import numpy as np
from functools import reduce
import types

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from util.helpers import safe_divide, hash_from_dict
from data_analysis.util.decorators import cache_to_file_decorator
from data_analysis.data_transformation.helpers import csunghye_logic
from data_analysis.data_transformation.helpers.linguistic_features_calculator import LinguisticFeatureLiteratureCalculator

class LinguisticFeatures(DataTransformer):
    """
    Linguistic features based on Dementia work
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Linguistic Features"
        print(f"Initializing {self.name}")

        # stanza nlp pipeline
        self.nlp_pipeline = stanza.Pipeline('en')

        # feature groups: which linguistic features to load
        valid_feature_groups = ['stanza_features',
                                'literature_features',
                                'csunghye_features']
        try:
            self.feature_groups = self.config.config_linguistic_features.feature_groups
        except (AttributeError, KeyError):
            self.feature_groups = ['literature_features', 'csunghye_features']
        assert all([fg in valid_feature_groups for fg in self.feature_groups])

        valid_feature_versions = ['full', 'reduced']
        try:
            self.feature_version = self.config.config_linguistic_features.feature_version
        except (AttributeError, KeyError):
            self.feature_version = 'full'
        assert self.feature_version in valid_feature_versions, \
            f"Invalid feature version {self.feature_version}, should be in {valid_feature_versions}"

        try:
            self.selected_features = self.config.config_linguistic_features.selected_features
        except (AttributeError, KeyError):
            self.selected_features = []  # empty list -> select all

        print(f"... using linguistic feature_groups {self.feature_groups}, feature_version {self.feature_version}, "
              f"selected_features {self.selected_features}")

    @cache_to_file_decorator(n_days=30)
    def calculate_stanza_feature_for_text(self, text):
        # Calculate pronoun / noun and verb / noun ratios
        doc = self.nlp_pipeline(text)
        counts = defaultdict(lambda: 0)
        for sentence in doc.sentences:
            for word in sentence.words:
                counts[word.pos] += 1

        n_coordinate_clauses, n_subordinate_clauses, fraction_coordinate_subordinate = \
            self._count_coordinate_subordinate_clauses(doc)

        n_words = len([word for sent in doc.sentences for word in sent.words])

        features_text = {
            'pronoun_noun_ratio': safe_divide(counts['PRON'], counts['NOUN']),
            'pronoun_pronounnoun_ratio': safe_divide(counts['PRON'], (counts['NOUN'] + counts['PRON'])),
            # according to https://content.iospress.com/articles/journal-of-alzheimers-disease/jad220847
            'verb_noun_ratio': safe_divide(counts['VERB'], counts['NOUN']),
            'n_words_stanza': n_words,
            'coordinate_clauses_per_word': safe_divide(n_coordinate_clauses, n_words),
            'subordinate_clauses_per_word': safe_divide(n_subordinate_clauses, n_words),
            'fraction_coordinate_subordinate': fraction_coordinate_subordinate
        }
        return features_text

    def _calculate_stanza_features(self, dataset):
        """
        Using Stanza NLP library
        """

        features = [self.calculate_stanza_feature_for_text(text) if not pd.isna(text) else {} for text in dataset.transcripts]

        return features

    def _count_coordinate_clauses(self, tree):
        # current node is S and has at least 2 child notes with also S => These two are coordinate clauses
        this_node_is_coordination = tree.label == 'S' and len([c for c in tree.children if c.label == 'S']) >= 2
        is_coordination = 1 if this_node_is_coordination else 0

        # add coordination here to all present in children, recursively
        return is_coordination + sum([self._count_coordinate_clauses(c) for c in tree.children])

    def _count_subordinate_clauses(self, tree):
        # current node has SBAR (subordinate clause) tagset
        is_subordination = 1 if tree.label == 'SBAR' else 0
        return is_subordination + sum([self._count_subordinate_clauses(c) for c in tree.children])

    def _count_coordinate_subordinate_clauses(self, doc):
        """
        Note that this logic is somewhat flawed, or does not really give what we want.
        Check out the test in this class for some observed problematic cases.
        Also have a look at https://github.com/jheitz/dementia/blob/main/src/scripts/coordinate_subordinate_clauses.py
        to see how it behaves on all transcripts
        """
        n_coordinate_clauses = 0
        n_subordinate_clauses = 0
        for sentence in doc.sentences:
            tree = sentence.constituency
            n_coordinate_clauses += self._count_coordinate_clauses(tree)
            n_subordinate_clauses += self._count_subordinate_clauses(tree)

        fraction = (n_coordinate_clauses + 1) / (n_subordinate_clauses + 1)
        return n_coordinate_clauses, n_subordinate_clauses, fraction

    @cache_to_file_decorator(n_days=30)
    def _calculate_literature_features_for_text(self, text):
        doc = self.nlp_pipeline(text.lower())

        feature_calculator = LinguisticFeatureLiteratureCalculator(doc, constants=self.CONSTANTS)
        count_pos, count_xpos = feature_calculator.pos_counts()
        n_words = feature_calculator.n_words()

        # todo: Use sigmoid_fraction for e.g. pronoun-noun-ratio, since it's symmetrical and actually makes the
        # features more expressive. we don't do it right now to stay consistent with prior literature.
        def sigmoid_fraction(a, b):
            assert a >= 0 and b >= 0
            a = a + 0.001 if a == 0 else a
            b = b + 0.001 if b == 0 else b
            return 1 / (1 + np.exp(-np.log(a / b)))

        features_pos = {
            'pronoun_noun_ratio': safe_divide(count_pos['PRON'], count_pos['NOUN']),  # [1], [2], [4]
            'verb_noun_ratio': safe_divide(count_pos['VERB'], count_pos['NOUN']),  # [4]
            'subordinate_coordinate_conjunction_ratio': safe_divide(count_pos['SCONJ'], count_pos['CCONJ']),  # [3]
            'adverb_ratio': safe_divide(count_pos['ADV'], n_words),  # [1], [2], [9]
            'noun_ratio': safe_divide(count_pos['NOUN'], n_words),  # [1], [8], [9]
            'verb_ratio': safe_divide(count_pos['VERB'], n_words),  # [1], [9]
            'pronoun_ratio': safe_divide(count_pos['PRON'], n_words),  # [2], [9]
            'personal_pronoun_ratio': safe_divide(count_xpos['PRP'], n_words),  # [2]
            'determiner_ratio': safe_divide(count_pos['DET'], n_words),  # [8]
            'preposition_ratio': safe_divide(count_xpos['IN'], n_words),  # [9]
            'verb_present_participle_ratio': safe_divide(count_xpos['VBG'], n_words),  # [2, 8]
            'verb_modal_ratio': safe_divide(count_xpos['MD'], n_words),  # [8]
            'verb_third_person_singular_ratio': safe_divide(count_xpos['VBZ'], n_words), # [1] (I suppose by inflected verbs they mean 3. person)

            'interjection_ratio': safe_divide(count_pos['INTJ'], n_words),  # Proxy for hestiation markers - INTJ are almost exclusively hestiations - Suggested by gschneid
            #'adjective_ratio': safe_divide(count_pos['ADJ'], n_words),  # Suggested by gschneid, removed again, because this produces a linear dependence to content_density
        }

        constituency_rules_count = feature_calculator.constituency_rules_count()

        constituents = feature_calculator.get_constituents()
        NP = [c for c in constituents if c['label'] == 'NP']
        PP = [c for c in constituents if c['label'] == 'PP']
        VP = [c for c in constituents if c['label'] == 'VP']
        PRP = [c for c in constituents if c['label'] == 'PRP']

        features_constituency = {
            # NP -> PRP means "count the number of noun phrases (NP) that consist of a pronoun (PRP)"
            'NP -> PRP': constituency_rules_count['NP -> PRP'],  # [1]
            'ADVP -> RB': constituency_rules_count['ADVP -> RB'],  # [1], [2]
            'NP -> DT_NN': constituency_rules_count['NP -> DT_NN'],  # [1]
            'ROOT -> FRAG': constituency_rules_count['ROOT -> FRAG'],  # [1]
            'VP -> AUX_VP': constituency_rules_count['VP -> AUX_VP'],  # [1]
            'VP -> VBG': constituency_rules_count['VP -> VBG'],  # [1]
            'VP -> VBG_PP': constituency_rules_count['VP -> VBG_PP'],  # [1]
            'VP -> IN_S': constituency_rules_count['VP -> IN_S'],  # [1]
            'VP -> AUX_ADJP': constituency_rules_count['VP -> AUX_ADJP'],  # [1]
            'VP -> AUX': constituency_rules_count['VP -> AUX'],  # [1]
            'VP -> VBD_NP': constituency_rules_count['VP -> VBD_NP'],  # [1]
            'INTJ -> UH': constituency_rules_count['INTJ -> UH'],  # [1]
            'NP_ratio': safe_divide(len(NP), len(constituents)),  # [9]
            'PRP_ratio': safe_divide(len(PRP), len(constituents)),  # [9]
            'PP_ratio': safe_divide(len(PP), len(constituents)),  # [1]
            'VP_ratio': safe_divide(len(VP), len(constituents)),  # [1]
            'avg_n_words_in_NP': safe_divide(sum([len(c['text'].split(" ")) for c in NP]), len(NP)),  # [9]
        }

        simple_features = {
            'n_words': n_words,  # [9], [4], [6], [8]
            'n_unique_words': feature_calculator.n_unique(),  # [6], [8]
            'avg_word_length': feature_calculator.word_length(),  # [1], [2]
            'avg_sentence_length': feature_calculator.sentence_length(),  # [4]
            'words_not_in_dict_ratio': feature_calculator.not_in_dictionary(),  # [1], [2]
        }

        vocabulary_richness_features = {
            'brunets_index': feature_calculator.brunets_index(),  # [3], [8]
            'honores_statistic': feature_calculator.honores_statistic(),  # [1], [9], [3], [8]
            'ttr': feature_calculator.ttr(),  # [4], [8]
            'mattr': feature_calculator.mattr(),  # [8]
            'NDW-50': feature_calculator.ndw50(),
        }

        readability_features = {
            'flesch_kincaid': feature_calculator.flesch_kincaid()  # [3]
        }

        avg_distance_between_utterances, prop_dist_thresh_05 = feature_calculator.cos_dist_between_sentences()
        repetitiveness_features = {
            'avg_distance_between_utterances': avg_distance_between_utterances,  # [1], [2]
            'prop_utterance_dist_below_05': prop_dist_thresh_05  # [1], [2]
        }

        density_features = {
            'propositional_density': safe_divide(
                count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'] + count_pos['ADP'] +
                count_pos['CCONJ'] + count_pos['SCONJ'], n_words),  # [3], [7]
            'content_density': safe_divide(
                count_pos['NOUN'] + count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'], n_words),
            # [3], [8], [9]
        }

        all_features = {**features_pos, **features_constituency, **simple_features, **vocabulary_richness_features,
                        **readability_features, **repetitiveness_features, **density_features}

        # let's drop the null feature (0 in every sample in ADReSS):
        all_features = {f: all_features[f] for f in all_features if
                        f not in ['VP -> AUX_VP', 'VP -> IN_S', 'VP -> AUX_ADJP', 'VP -> AUX']}

        # rename features
        all_features = {f"lit_{f}": all_features[f] for f in all_features}

        return all_features

    def _load_literature_features(self, dataset):
        features = [self._calculate_literature_features_for_text(text) if not pd.isna(text) else {} for text in dataset.transcripts]

        if self.feature_version == 'reduced':
            # let's drop some features to remove strong correlations (tested in pictureDescription/google train samples)
            # see /analyses/kw49/reduced_features/feature_correlations.ipynb
            features_to_remove = [
                'lit_avg_sentence_length',  # 0.99 with flesch_kincaid
                'lit_PRP_ratio',  # 0.99 with personal_pronoun_ratio
                'lit_ttr',  # 0.99 with brunets_index
                'lit_NP -> DT_NN',  # 0.95 with n_words
                'lit_pronoun_noun_ratio',  # 0.93 with pronoun_ratio
                'lit_n_unique_words',  # 0.91 with n_words
                'lit_ADVP -> RB',  # 0.91 with n_words
                'lit_PP_ratio',  # 0.89 with preposition ratio
            ]
            assert all([f in features[0].keys() for f in features_to_remove]), f"{[f for f in features_to_remove if f not in features[0].keys()]} vs {features[0].keys()}"
            features = [{f: features_one[f] for f in features_one if f not in features_to_remove} for features_one in features]

        # print("Extracted literature features:", [f for f in features[0].keys()])

        return features

    def _print_parse_trees_and_pos_tags(self, dataset: Dataset):
        for sample_name, text in zip(dataset.sample_names, dataset.transcripts):
            if pd.isna(text):
                continue

            doc = self.nlp_pipeline(text.lower())
            parse_trees = [sentence.constituency.pretty_print() for sentence in doc.sentences]

            parse_tree_base_dir = os.path.join("/Users/jheitz/git/luha-prolific-study/analyses/2024/kw46/luha_parse_trees")
            os.makedirs(parse_tree_base_dir, exist_ok=True)
            with open(os.path.join(parse_tree_base_dir, f"cookieTheft_{sample_name}.txt"), "w") as f:
                f.write(text + "\n\n")
                f.write("\n".join(parse_trees))

            pos_annotation = lambda sentence: " ".join([f"{word.text} [{word.pos}]" for word in sentence.words])
            text_pos_annotated = "\n".join([pos_annotation(sent) for sent in doc.sentences])
            pos_tags_base_dir = os.path.join("/Users/jheitz/git/luha-prolific-study/analyses/2024/kw46/luha_pos_tags")
            os.makedirs(pos_tags_base_dir, exist_ok=True)
            with open(os.path.join(pos_tags_base_dir, f"cookieTheft_{sample_name}.txt"), "w") as f:
                f.write(text_pos_annotated)

            pos_annotation = lambda sentence: " ".join([f"{word.text} [{word.xpos}]" for word in sentence.words])
            text_pos_annotated = "\n".join([pos_annotation(sent) for sent in doc.sentences])
            pos_tags_base_dir = os.path.join("/Users/jheitz/git/luha-prolific-study/analyses/2024/kw46/luha_xpos_tags")
            os.makedirs(pos_tags_base_dir, exist_ok=True)
            with open(os.path.join(pos_tags_base_dir, f"cookieTheft_{sample_name}.txt"), "w") as f:
                f.write(text_pos_annotated)

    @cache_to_file_decorator(n_days=30)
    def _load_csunghye_features_one(self, text, static_hash):
        """
        Run the logic for one text
        'static_hash' parameter is used for the caching decorator -
        if anything changed in the static data, the features would be recomputed
        """
        print(f"Calculating csunghye features for text {text[:50]}")

        # run first-pass pos tagging on the entire doc (POS tags are used to exclude repetitions)
        doc = self.nlp_pipeline(text)
        words = [word for sentence in doc.sentences for word in sentence.words]

        # clean the doc
        cleaned_text, total_word, um, uh, eh, hm, yeah, partial, repetition, restart = csunghye_logic.clean_doc(words)
        # tally the total word count + dysfluency markers
        total = total_word + um + uh + eh + hm + yeah + partial + repetition

        # run second-pass pos tagging on cleaned texts only (without dysfluency markers)
        doc = self.nlp_pipeline(cleaned_text)
        words = [word for sentence in doc.sentences for word in sentence.words]

        # rate lexical measures for each word
        lexical = csunghye_logic.attach_lexical(words, self.csunghye_static.measureDict, self.csunghye_static.phonDf)

        # calculate averaged lexical measure
        lexicalSumDF = csunghye_logic.lexical_summary(lexical)
        assert lexicalSumDF.shape[0] == 1

        features = lexicalSumDF.iloc[0].to_dict()

        return features

    def _load_csunghye_features(self, dataset):
        # get lexical measures to use
        lexical_lookup_filepath = os.path.join(self.CONSTANTS.RESOURCES_DIR, 'csunghye_all_measures_raw.csv')
        self.csunghye_static = types.SimpleNamespace(
            measureDict=pd.read_csv(lexical_lookup_filepath),
            phonDf=csunghye_logic.phondict(),
        )
        csunghye_static_hash = hash_from_dict(self.csunghye_static.__dict__)

        features = [self._load_csunghye_features_one(text, csunghye_static_hash) if not pd.isna(text) else {} for text in dataset.transcripts]

        if self.feature_version == 'reduced':
            # let's only keep the _content version of the features (not _all / _noun), as they are highly correlated
            # the content version appears to be most useful (highest effect size in pictureDescription/google/train)
            # Also the syll / phon features are already captured by similar metrics (hightly correlated to n_words)
            # see /analyses/kw49/reduced_features/feature_correlations.ipynb / /analyses/kw49/reduced_features/csunghye_effect_sizes.ipynb
            # update 6.1.25: We keep the noun version instead of content version, because that's the one published in https://www.sciencedirect.com/science/article/pii/S001094522100037X
            print("Keeping only reduced set of csunghye features")
            features_to_keep = ['AoA_noun', 'frequency_noun', 'familiarity_noun', 'ambiguity_noun', 'concreteness_noun']
            assert all([f in features[0].keys() for f in features_to_keep]), f"Some of {features_to_keep} not in {features[0].keys()}"
            features = [{f: features_one[f] for f in features_one if f in features_to_keep} for features_one in features]

        # rename features
        features = [{f"sung_{f}": features_one[f] for f in features_one} for features_one in features]

        # print("Extracted csunghye features:", [f for f in features[0].keys()])

        return features

    def _load_features(self, dataset: Dataset):
        assert isinstance(dataset, Dataset), "Input should be Dataset"
        assert 'transcripts' in dataset.data_variables and isinstance(dataset.transcripts, np.ndarray) and len(dataset.transcripts.shape) == 1, \
            "Dataset does not have a valid transcript data variable, which is necessary to calculate the linguistic features on"

        features_collected = []
        for feature_group in self.feature_groups:
            if feature_group == 'stanza_features':
                features_collected.append(self._calculate_stanza_features(dataset))
            elif feature_group == 'literature_features':
                features_collected.append(self._load_literature_features(dataset))
            elif feature_group == 'csunghye_features':
                features_collected.append(self._load_csunghye_features(dataset))
            else:
                raise ValueError(f"Invalid feature group {feature_group}")

        # combine multiple (feature group) dictionaries of {feature_name: value} for each sample,
        # getting one dictionary per sample
        all_features = [reduce(lambda a, b: {**a, **b}, sample) for sample in zip(*features_collected)]

        if False:
            self._print_parse_trees_and_pos_tags(dataset)

        # select specific subset of features
        if len(self.selected_features) > 0:
            non_existing = [f for f in self.selected_features if f not in all_features[0].keys()]
            if len(non_existing) > 0:
                print("Warning (Linguistic features): the following features requested in the config file have do not exist:", non_existing)
            all_features = [{feature: sample[feature] for feature in sample if feature in self.selected_features} for sample in all_features]

        features_df = pd.DataFrame(all_features)

        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config

        new_dataset.features = self._create_new_feature_df(new_dataset, features_df)

        return new_dataset


    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Calculating linguistic features for dataset {dataset}")
        return self._load_features(dataset)


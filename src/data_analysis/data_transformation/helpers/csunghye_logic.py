# original source: https://github.com/csunghye/lexical_pipeline/tree/main
import pandas as pd
import nltk

# lists of predefined words
FILLERS = ['um', 'uh', 'eh']
BACKCHANNELS = ['hm', 'yeah', 'mhm', 'huh']
EXCEPTION_RULES = ["that 's that", "that is that", "as well as", "as much as", "as ADJ as", "the NOUN the", "do n't do"]
PUNCT = ['.', '?', "'"]



# update the list of previous words to count repetitions
def update_words(prev_bi, prev, word):
	penult_bi = prev_bi
	prev_bi = prev+word.text.lower()
	penult = prev
	prev = word.text.lower()
	prev_pos = word.pos
	return penult_bi, prev_bi, penult, prev, prev_pos


# count fillers (uses previously defined list)
def count_fillers(word, um, uh, eh):
	if word in FILLERS:
		if word == "um":
			um += 1
		elif word == "uh":
			uh += 1
		else:
			eh += 1
	else:
		pass

# count backchannels (uses previously defined list)
def count_backchannels(word, hm, yeah):
	if word in BACKCHANNELS:
		if word == "hm":
			hm += 1
		elif word == "yeah":
			yeah += 1
		else:
			pass
	else:
		pass



# clean the transcripts by excluding dysfluency markers and repetitions => This helps improving the pos tagging accuracy.
def clean_doc(doc):
    # initiate dysfluency marker counts
    um = uh = eh = hm = yeah = partial = restart = 0
    repetitionList = []

    # initiate previous words
    prev_bi = penult_bi = prev = penult = prev_pos = 'NA'

    # list of cleaned words to be used for second-pass pos tagging
    cleaned = []

    # loope through words in a doc
    for word in doc:
        if word.pos != "SPACE" and word.text != ",":
            # count dysfluency markers
            if (word.text.lower() in FILLERS) or (word.text.lower() in BACKCHANNELS) or word.text.endswith(
                    '-') or word.text.endswith('=') or (word.text == "#"):
                count_fillers(word.text.lower(), um, uh, eh)
                count_backchannels(word.text.lower(), hm, yeah)
                if word.text.endswith('-'):
                    partial += 1
                elif word.text.endswith('='):
                    repetitionList.append(word.text)
                elif word.text == '#':
                    restart += 1

            else:
                # remove repetitions
                if word.text.lower() != prev:  # check if a given word is a repetition of a previous word (e.g., this, <this> boy is ...)
                    if word.text.lower() != penult:  # check if a given word is a repetition of a preceding word of the previous word (e.g., this, uh, <this> boy ... )
                        # if a phrase is not repeated, do not count as a repetition (e.g., This boy, this girl is ... )
                        if prev_bi != prev.lower() + word.text.lower() and penult_bi != prev.lower() + word.text.lower():
                            cleaned.append(word.text)
                            penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)
                        else:
                            cleaned.pop(-1)
                    # do not count as repetition if a phrase is in the exception rules (e.g., as soon <as> possible)
                    elif penult + " " + prev + " " + word.text.lower() in EXCEPTION_RULES or penult + " " + prev_pos + " " + word.text.lower() in EXCEPTION_RULES:
                        cleaned.append(word.text)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)
                    # if the preceding word is a sentence boundary, do not count as a repetition (e.g., She ate this. <This> boy is ...)
                    elif word.text.lower() == penult and (prev == "." or prev == "?"):
                        cleaned.append(word.text)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)
                    # pass repetition of punctuation marks
                    elif word.text in PUNCT and penult in PUNCT:
                        pass
                    # if a given word is a sentence boundary, do not count as a repetition
                    elif word.text == "." or word.text == "?":
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)

                    else:  # if a preceding word of a previous word is a copy of the given word, count it as a repetition
                        repetitionList.append(word)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)

                elif word.text in PUNCT and prev in PUNCT:
                    pass

                elif word.text == "mm" and prev == "mm":
                    cleaned.pop(-1)
                elif word.text == "mhm" and prev == "mhm":
                    cleaned.pop(-1)

                else:  # if a previous word is an exact copy of the given word, count it as a repetition
                    repetitionList.append(word)
                    penult_bi, prev_bi, penult, prev, prev_pos = update_words(prev_bi, prev, word)

    return ' '.join(cleaned), len(cleaned), um, uh, eh, hm, yeah, partial, len(repetitionList), restart


def attach_lexical(document, measureDict, phonDf):
    # make a df from a cleaned text
    words = pd.DataFrame({'word': [w.text.lower() for w in document], 'lemma': [w.lemma for w in document],
                          'pos': [w.pos for w in document]})
    # remove punctuation marsk
    words = words[words.pos != "PUNCT"]
    # rate lexical measures based on word and lemma
    word_lexical = pd.merge(words, measureDict, on='word', how='left')
    lemma_lexical = pd.merge(words[["lemma", "pos"]], measureDict, left_on='lemma', right_on='word', how='left')
    # if word-level lexical measures are not available, use lemma-based lexical measures
    df = word_lexical.fillna(lemma_lexical)
    # merge with the phone df
    df = pd.merge(df, phonDf[["word", "phon", "syll"]], on='word', how='left')

    # fill out phoneme and syllable counts for contracted words (not included in the CMU pronunciation dict)
    df.loc[df.word == "'s", ["phon", "syll"]] = df.loc[df.word == "'ve", ["phon", "syll"]] = df.loc[
        df.word == "'ll", ["phon", "syll"]] = df.loc[df.word == "'m", ["phon", "syll"]] = df.loc[
        df.word == "'d", ["phon", "syll"]] = df.loc[df.word == "'re", ["phon", "syll"]] = [1, 0]
    df.loc[df.word == "n't", ["phon", "syll"]] = [2, 0]
    df.loc[df.word == "gon", ["phon", "syll"]] = df.loc[df.word == "wan", ["phon", "syll"]] = [3,
                                                                                               1]  # as in gonna or wanna
    df.loc[df.word == "wo", ["phon", "syll"]] = df.loc[df.word == "na", ["phon", "syll"]] = [2,
                                                                                             1]  # as in won't or wanna

    return df


def lexical_summary(lexical_df):
    # get mean values of the rated lexical measures of all words in a given df
    allwords = pd.DataFrame(lexical_df.iloc[:, 3:].mean()).transpose()
    # rename columns
    allwords.columns = ['frequency_all', 'AoA_all', 'familiarity_all', 'ambiguity_all', 'concreteness_all', 'phone_all',
                        'syll_all']
    # count total number of syllables from all words
    total_syll = lexical_df.iloc[:, -1].sum()

    # get mean values of the lexical measures of all nouns
    nouns = lexical_df[lexical_df.pos == 'NOUN']
    nouns_mean = pd.DataFrame(nouns.iloc[:, 3:].mean()).transpose()
    nouns_mean.columns = ['frequency_noun', 'AoA_noun', 'familiarity_noun', 'ambiguity_noun', 'concreteness_noun',
                          'phone_noun', 'syll_noun']

    # get mean values of the lexical measures of all content words
    content = lexical_df[
        (lexical_df.pos == 'NOUN') | (lexical_df.pos == "ADV") | (lexical_df.pos == "ADJ") | (lexical_df.pos == "VERB")]
    content_mean = pd.DataFrame(content.iloc[:, 3:].mean()).transpose()
    content_mean.columns = ['frequency_content', 'AoA_content', 'familiarity_content', 'ambiguity_content',
                            'concreteness_content', 'phone_content', 'syll_content']

    # combine all lexical measure dfs
    all_df = pd.concat([allwords, nouns_mean], axis=1)
    all_df2 = pd.concat([all_df, content_mean], axis=1)
    all_df2['total_syll'] = total_syll

    return all_df2


def phondict():
    # calculate number of phonemes and syllables using the CMU pronouncing dict
    nltk.download('cmudict')
    # get the CMU dict as a dataframe
    phonDf = pd.DataFrame.from_dict(nltk.corpus.cmudict.dict(), orient='index')
    phonDf = phonDf.reset_index()
    # count number of phonemes and make it as a column of df
    phonDf['phon'] = phonDf[0].map(len)
    # drop unnecessary columns and clean the df
    phonDf = phonDf.drop(columns=[1, 2, 3, 4])
    phonDf.columns = ['word', 'pron', 'phon']
    # comma-join the phonemes and count syllables
    phonDf['pronstring'] = [','.join(map(str, l)) for l in phonDf['pron']]
    phonDf['syll'] = phonDf.pronstring.str.count("0|1|2")

    return phonDf
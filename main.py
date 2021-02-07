"""
CSC2611 Lab
Construct various language models and investigate word pair similarity
Wendy Qiu 2021.02.01
"""

import wordpairs
import numpy as np
from gensim.models import KeyedVectors
from os.path import join
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pickle

LOCAL_DIR = 'C:/Users/Ronghui/Documents/csc2611/lab/'


def find_most_similar(word_ukn):
    pass


def part_one():
    """Part 1: Synchronic word embedding"""

    # step 1 & 2: get word embeddings for word pairs
    embedding_path = join(LOCAL_DIR, 'embeddings/googleNews/GoogleNews-vectors-negative300.bin')
    model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    table_word_embeddings = {}
    for word in wordpairs.table_word_set:
        curr_vec = model[word]
        table_word_embeddings[word] = curr_vec

    # step 3: calculate cosine distance
    similarity_list = []
    for each_pair in wordpairs.P_word_pairs:
        a_vec = table_word_embeddings[each_pair[0]]
        b_vec = table_word_embeddings[each_pair[1]]
        similarity = cosine_similarity(np.array(a_vec).reshape(1, -1), np.array(b_vec).reshape(1, -1))[0][0]
        similarity_list.append(similarity)

    # step 3 continued: calculate Pearson correlation
    corr, _ = pearsonr(np.array(similarity_list), np.array(wordpairs.S_word_pair_score))
    print('Pearson correlation between word2vec-based & human similarities: {0}'.format(corr))

    # unpickle the LSA-300 embedding
    filename = join(LOCAL_DIR, 'embeddings/exercise/df_m2_300')
    infile = open(filename, 'rb')
    df_M2_300 = pickle.load(infile)
    infile.close()

    # step 4: analogy test on pre-trained word2vec embeddings vs. LSA (300 dimensions)
    test_path = join(LOCAL_DIR, 'test/word-test_v1.txt')

    word2vec_correct_count = 0
    lsa_correct_count = 0
    test_counter = 0

    full_test_set = []
    with open(test_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        total_counter = len(lines)
        for line in lines:
            if not line.startswith(('/', ':')):
                curr_test_set = line.split()
                full_test_set.append(curr_test_set)

                # first do analogy test on M2_300, if possible, do word2vec next
                if ((df_M2_300.index == curr_test_set[0]).any() or (df_M2_300.index == curr_test_set[1]).any() or
                        (df_M2_300.index == curr_test_set[2]).any()):
                    test_counter += 1
                    word_1 = df_M2_300[curr_test_set[0]]
                    word_2 = df_M2_300[curr_test_set[1]]
                    word_3 = df_M2_300[curr_test_set[2]]
                    print("type of word_1: {0}".format(type(word_1)))
                    print("word_1: ")
                    print(word_1)
                    word_ukn = word_2 + word_3 - word_1
                    # find the most similar word
                    most_similar = 0.
                    for index, row in df_M2_300.iterrows():
                        print(row)
                        curr_similarity = \
                            cosine_similarity(np.array(row).reshape(1, -1), np.array(word_ukn).reshape(1, -1))[0][0]
                        if curr_similarity > most_similar:
                            most_similar = curr_similarity
                            prediction_lsa = index
                            print(prediction_lsa)
                    if prediction_lsa == curr_test_set[3]:
                        lsa_correct_count += 1
                        print("LSA correct")

                    # analogy test on word2vec
                    prediction_word2vec = model.most_similar(negative=[curr_test_set[0]],
                                                    positive=[curr_test_set[1], curr_test_set[2]])[0][0]
                    if prediction_word2vec == curr_test_set[3]:
                        word2vec_correct_count += 1
                        print("word2vec correct")

                    print("lsa prediction is: {0}, correct answer is: {1}".format(prediction_lsa, curr_test_set[3]))
                    print("word2vec prediction is: {0}".format(prediction_word2vec))

    print("The total number of possible test cases is {0}.\n{1} of them are feasible tests"
          .format(total_counter, test_counter))
    print("The accuracy for LSA is {0}, and word2vec is {1}"
          .format(lsa_correct_count/test_counter, word2vec_correct_count/test_counter))


def part_two():
    """Part 2: Diachronic word embedding"""
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("executing part one function...")
    part_one()
    print("executing part two function...")
    part_two()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

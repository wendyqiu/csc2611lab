"""
CSC2611 Lab
Construct various language models and investigate word pair similarity
Wendy Qiu 2021.02.01
"""

import wordpairs
import numpy as np
from gensim.models import KeyedVectors
from os.path import join, exists
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
import pickle
from prettytable import PrettyTable

LOCAL_DIR = 'C:/Users/Ronghui/Documents/csc2611/lab/'


def pickle_save(file_path, file):
    print("saving python pickles at: {}".format(file_path))
    outfile = open(file_path, 'wb')
    pickle.dump(file, outfile)
    outfile.close()


def load_pickle(file_path):
    print("loading python pickles at: {}".format(file_path))
    infile = open(file_path, 'rb')
    result = pickle.load(infile)
    infile.close()
    return result


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
    print('Step 3: Pearson correlation between word2vec-based & human similarities: {0}'.format(corr))

    # unpickle the LSA-300 embedding
    filename = join(LOCAL_DIR, 'embeddings/exercise/df_m2_300_0.038')
    df_M2_300 = load_pickle(filename)

    # step 4: analogy test on pre-trained word2vec embeddings vs. LSA (300 dimensions)
    # test_path = join(LOCAL_DIR, 'test/test.txt')
    test_path = join(LOCAL_DIR, 'test/word-test_v1.txt')

    word2vec_correct_count = 0
    lsa_correct_count = 0
    lsa_correct_count_top3 = 0
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
                if ((df_M2_300.index == curr_test_set[0].lower()).any() and
                        (df_M2_300.index == curr_test_set[1].lower()).any() and
                        (df_M2_300.index == curr_test_set[2].lower()).any() and
                        (df_M2_300.index == curr_test_set[3].lower()).any()):
                    print("entering now: {}".format(curr_test_set))

                    test_counter += 1
                    word_ukn_df = df_M2_300.loc[curr_test_set[1].lower()] + \
                                                df_M2_300.loc[curr_test_set[2].lower()] - \
                                                df_M2_300.loc[curr_test_set[0].lower()]
                    word_ukn = word_ukn_df.to_numpy()
                    # find the most similar word
                    most_similar = 0.
                    prediction_lsa = None
                    second_choice = None
                    third_choice = None
                    for index, row in df_M2_300.iterrows():
                        curr_similarity = \
                            cosine_similarity(row.to_numpy().reshape(1, -1), word_ukn.reshape(1, -1))[0][0]
                        if curr_similarity > most_similar and index != curr_test_set[0].lower() and \
                                index != curr_test_set[1].lower() and index != curr_test_set[2].lower():
                            most_similar = curr_similarity
                            prediction_lsa = index
                            second_choice = prediction_lsa
                            third_choice = second_choice
                    if prediction_lsa == curr_test_set[3].lower():
                        lsa_correct_count += 1
                    if second_choice == curr_test_set[3].lower() or third_choice == curr_test_set[3].lower():
                        lsa_correct_count_top3 += 1

                    # analogy test on word2vec
                    prediction_word2vec = model.most_similar(negative=[curr_test_set[0]],
                                                             positive=[curr_test_set[1], curr_test_set[2]])[0][0]
                    if prediction_word2vec == curr_test_set[3]:
                        word2vec_correct_count += 1

                    print("lsa prediction is: {0}, correct answer is: {1}".format(prediction_lsa, curr_test_set[3]))
                    print("word2vec prediction is: {0}".format(prediction_word2vec))

    print("The total number of possible test cases is {0}.\n{1} of them are feasible tests"
          .format(total_counter, test_counter))
    print("The accuracy for LSA is {0}, and word2vec is {1}"
          .format(lsa_correct_count / test_counter, word2vec_correct_count / test_counter))
    print("The top-3 accuracy for LSA is {}".format(lsa_correct_count_top3 / test_counter))


def basic_cosine_sim(old_embedding, new_embedding):
    """calculate the absolute cosine similarity of each word between 2 time periods
       the semantic change is defined as the inverse of similarity score
       (smaller score = larger change)"""
    all_sim_rank = []
    for i in range(len(old_embedding)):
        sim = cosine_similarity(np.array(old_embedding[i]).reshape(1, -1), np.array(new_embedding[i]).reshape(1, -1))[0][0]
        all_sim_rank.append([sim, i])
    all_sim_rank.sort(key=lambda x: x[0])
    top20 = all_sim_rank[:20]
    least20 = all_sim_rank[:-21:-1]
    return [x[1] for x in top20], [x[1] for x in least20], [x[1] for x in all_sim_rank]


def k_cluster(old_embedding, new_embedding, k=5):
    """find top k nearest neighbours and compute their differences b/w 2 time periods
       semantic changes are defined as how much a word's k nearest neighbours shift between the periods
       (larger shifting average = larger change)"""
    full_sim_rank = []
    for i in range(len(old_embedding)):     # i is the focus word
        old_sim_list = []
        new_sim_list = []
        curr = np.array(old_embedding[i].reshape(1, -1))
        # find the top-k neighbor words for each focus word
        for j in range(len(old_embedding)):
            if i != j:
                curr_old = np.array(old_embedding[j].reshape(1, -1))
                old_sim = cosine_similarity(curr_old, curr)[0][0]
                old_sim_list.append([old_sim, j])
                curr_new = np.array(new_embedding[j].reshape(1, -1))
                new_sim = cosine_similarity(curr_new, curr)[0][0]
                new_sim_list.append([new_sim, j])
        # For both models, get a similarity vector between the focus word and top-k neighbor words
        new_sim_list.sort(key=lambda x: x[0], reverse=True)
        old_sim_list.sort(key=lambda x: x[0], reverse=True)
        closest_new_neighbour = new_sim_list[:k]
        closest_old_neighbour = old_sim_list[:k]
        meta_neighbor_idx = list(set(x[1] for x in closest_new_neighbour) | set(y[1] for y in closest_old_neighbour))

        vec1 = [cosine_similarity(curr, np.array(old_embedding[idx].reshape(1, -1)))[0][0] for idx in meta_neighbor_idx]
        vec2 = [cosine_similarity(curr, np.array(new_embedding[idx].reshape(1, -1)))[0][0] for idx in meta_neighbor_idx]

        # Compute the cosine distance between those similarity vectors:
        # a measure of the relative semantic shift for this word between these two models
        dist = cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]
        full_sim_rank.append([dist, i])
    full_sim_rank.sort(key=lambda x: x[0])
    print("full_distance_rank: ".format(full_sim_rank))
    top20 = full_sim_rank[:20]
    least20 = full_sim_rank[:-21:-1]
    return [x[1] for x in top20], [x[1] for x in least20], [x[1] for x in full_sim_rank]


def rank_all(old_embedding, new_embedding):
    """for each word in the embedding, select it as the origin, create 2 ranking lists (one for each time period)
       that include all other the words in the embedding, based on the cosine distance of a word to the origin
       compute the Spearman's rank correlation coefficient of these 2 ranking list
       the amount of semantic change is defined as the inverse of the Spearman's value
       (the closer the value is to 0, the larger the change is)"""
    rank_score = []
    for i in range(len(old_embedding)):
        rank_old = []
        rank_new = []
        origin = np.array(old_embedding[i]).reshape(1, -1)
        for j in range(len(old_embedding)):
            if i == j:  # the origin itself
                rank_old.append([0., i])
                rank_new.append([0., i])
            else:
                curr_compare_old = np.array(old_embedding[j].reshape(1, -1))
                sim_old = cosine_similarity(origin, curr_compare_old)[0][0]
                rank_old.append([sim_old, j])
                curr_compare_new = np.array(new_embedding[j].reshape(1, -1))
                sim_new = cosine_similarity(origin, curr_compare_new)[0][0]
                rank_new.append([sim_new, j])
        rank_new.sort(key=lambda x: x[0])
        rank_old.sort(key=lambda x: x[0])
        coef, _ = spearmanr([x[1] for x in rank_new], [x[1] for x in rank_old])
        rank_score.append([coef, i])
    print("rank_score: {}".format(rank_score))
    sorted_rank_score = sorted(rank_score, key=lambda x: abs(x[0]))
    print("sorted_rank_score: {}".format(sorted_rank_score))
    least20 = sorted_rank_score[:20]
    top20 = sorted_rank_score[:-21:-1]
    return [x[1] for x in top20], [x[1] for x in least20], [x[1] for x in sorted_rank_score]


def read_test(test_path):
    full_test_set = []
    with open(test_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            curr_test_set = line.split()
            full_test_set.append(curr_test_set)
    return full_test_set




def evaluate_method(diachronic_word_list, ordered_list):
    """3 test files are created based on the <Statistically Significant Detection of Linguistic Change> paper
       each file contains several words that undergo semantic changes
       the p-value indicates the significance of the change (inverse)
    """
    test1_path = join(LOCAL_DIR, 'test/semantic_change_test1.txt')
    test_set_1 = read_test(test1_path)
    tests

    test2_path = join(LOCAL_DIR, 'test/semantic_change_test2.txt')
    test_set_2 = read_test(test2_path)

    test3_path = join(LOCAL_DIR, 'test/semantic_change_test3.txt')
    test_set_3 = read_test(test3_path)






def part_two():
    """Part 2: Diachronic word embedding"""

    # unpickle the LSA-300 embedding
    diachronic_emb_path = join(LOCAL_DIR, 'embeddings/embeddings_downloaded/data.pkl')
    diachronic_dict = load_pickle(diachronic_emb_path)
    # The file is a dictionary that contains the following entries:
    # 'w': a list of 2000 words, a subset of the English lexicon
    # 'd': a list of decades between 1900 and 2000
    # 'E': a 2000 by 10 by 300 list of list of vectors;
    # the (i,j)-th entry is a 300-dimensional vector for the i-th word in the j-th decade

    # get 2 sets of word embeddings for the first and last time periods
    old_embedding = []
    new_embedding = []
    for each_word in diachronic_dict['E']:
        old_embedding.append(each_word[0])
        new_embedding.append(each_word[9])

    # step 2: three methods to measure degree of semantic change for each word
    save_methods_dir = join(LOCAL_DIR, 'checkpoint')

    diachronic_word_list = diachronic_dict['w']
    if exists(join(save_methods_dir, 'basic_full_list.pkl')):
        basic_full_list = load_pickle(join(save_methods_dir, 'basic_full_list.pkl'))
    else:
        basic_cosine_top20, basic_cosine_least20, basic_full_list = basic_cosine_sim(old_embedding, new_embedding)
        print("basic_cosine_top20 indexes: {}".format(basic_cosine_top20))
        print("basic_cosine_least20 indexes: {}".format(basic_cosine_least20))
        basic_cosine_top_words = [diachronic_word_list[i] for i in basic_cosine_top20]
        basic_cosine_bot_words = [diachronic_word_list[i] for i in basic_cosine_least20]
        print("basic_cosine_top_words: {}".format(basic_cosine_top_words))
        print("basic_cosine_bot_words: {}".format(basic_cosine_bot_words))
        pickle_save(join(save_methods_dir, 'basic_full_list.pkl'), basic_full_list)

    if exists(join(save_methods_dir, 'cluster_full_list.pkl')):
        cluster_full_list = load_pickle(join(save_methods_dir, 'cluster_full_list.pkl'))
    else:
        k_cluster_top20, k_cluster_least20, cluster_full_list = k_cluster(old_embedding, new_embedding, k=5)
        print("k_cluster_top20 indexes: {}".format(k_cluster_top20))
        print("k_cluster_least20 indexes: {}".format(k_cluster_least20))
        cluster_top_words = [diachronic_word_list[i] for i in k_cluster_top20]
        cluster_bot_words = [diachronic_word_list[i] for i in k_cluster_least20]
        print("cluster_top_words: {}".format(cluster_top_words))
        print("cluster_bot_words: {}".format(cluster_bot_words))
        pickle_save(join(save_methods_dir, 'cluster_full_list.pkl'), cluster_full_list)

    if exists(join(save_methods_dir, 'rank_full_list.pkl')):
        rank_full_list = load_pickle(join(save_methods_dir, 'rank_full_list.pkl'))
    else:
        rank_top20, rank_least20, rank_full_list = rank_all(old_embedding, new_embedding)
        print("rank_top20 indexes: {}".format(rank_top20))
        print("rank_least20 indexes: {}".format(rank_least20))
        ranking_top_words = [diachronic_word_list[i] for i in rank_top20]
        ranking_bot_words = [diachronic_word_list[i] for i in rank_least20]
        print("ranking_top_words: {}".format(ranking_top_words))
        print("ranking_bot_words: {}".format(ranking_bot_words))
        pickle_save(join(save_methods_dir, 'rank_full_list.pkl'), rank_full_list)

    # step 2: Measure the inter-correlations (of semantic change in all words) among the three methods
    b_k = pearsonr(np.array(basic_full_list), np.array(cluster_full_list))
    b_r = pearsonr(np.array(basic_full_list), np.array(rank_full_list))
    k_r = pearsonr(np.array(cluster_full_list), np.array(rank_full_list))
    table_header = ['Methods', 'Basic Cosine', 'k-cluster', 'Full Ranking']
    corr_table = PrettyTable(table_header)
    self = pearsonr(np.array(basic_full_list), np.array(basic_full_list))
    corr_table.add_row(['Basic Cosine', [round(num, 3) for num in self], [round(num, 3) for num in b_k],
                        [round(num, 3) for num in b_r]])
    self = pearsonr(np.array(cluster_full_list), np.array(cluster_full_list))
    corr_table.add_row(['cluster', [round(num, 3) for num in b_k], [round(num, 3) for num in self],
                        [round(num, 3) for num in k_r]])
    self = pearsonr(np.array(rank_full_list), np.array(rank_full_list))
    corr_table.add_row(['Full Ranking', [round(num, 3) for num in b_r], [round(num, 3) for num in k_r],
                        [round(num, 3) for num in self]])
    print(corr_table)







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO: uncomment part one
    # print("executing part one function...")
    # part_one()
    print("executing part two function...")
    part_two()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

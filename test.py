import gensim

from TF import TF


def get_similar_words_str(w, model, topn=10):
    result_words = get_similar_words_list(w, model)
    return str(result_words)


def get_similar_words_list(w, model, topn=10):
    result_words = []
    try:
        similary_words = model.most_similar(w, topn=10)
        # print(similary_words)
        for (word, similarity) in similary_words:
            result_words.append(word)
        # print(result_words)
    except:
        print("There are some errors!" + w)

    return result_words


def load_models(model_path):
    return gensim.models.Word2Vec.load(model_path)

if "__name__ == __main__()":
    filename = "topics.txt"  # source file
    tf = TF(filename)
    keywords = tf.get_key_words()

    model_path = "information_model0830_pre_extra"  # model file
    model = load_models(model_path)

    for word, _ in keywords[1:]:
        words = get_similar_words_str(word, model)
        print(word, words)
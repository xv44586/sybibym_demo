import codecs
import jieba, jieba.analyse
import gensim
from gensim.models.word2vec import LineSentence


def read_source_file(source_file_name):
    try:
        file_reader = codecs.open(source_file_name, 'r', 'utf-8', errors="ignore")
        lines = file_reader.readlines()
        print("Read complete!")
        file_reader.close()
        return lines
    except:
        print("There are some errors while reading.")


def write_file(target_file_name, content):
    file_write = codecs.open(target_file_name, 'w+', 'utf-8')
    file_write.writelines(content)
    print("Write sussfully!")
    file_write.close()


def separate_word(filename, user_dic_file, separated_file):
    print("separate_word")
    lines = read_source_file(filename)
    # jieba.load_userdict(user_dic_file)
    stopkey = [line.strip() for line in codecs.open('stop_words.txt', 'r', 'utf-8').readlines()]

    output = codecs.open(separated_file, 'w', 'utf-8')
    num = 0
    for line in lines:
        num = num + 1
        if num % 10000 == 0:
            print("Processing line number: " + str(num))
        seg_word_line = jieba.cut(line, cut_all=True)
        # seg_word_line = jieba.analyse.extract_tags(line, 20)
        wordls = list(set(seg_word_line) - set(stopkey))
        if len(wordls) > 0:
            word_line = ' '.join(wordls) + '\n'
        output.write(word_line)
    output.close()
    return separated_file


def build_model(source_separated_words_file, model_path):
    print("start building...", source_separated_words_file)
    model = gensim.models.Word2Vec(LineSentence(source_separated_words_file), size=200, window=5, min_count=5,
                                   alpha=0.02, workers=4)
    model.save(model_path)
    print("build successful!", model_path)
    return model


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
    user_dic_file = "new_dict.txt"  # user dic file
    separated_file = "dk_spe_file_pre_extra.txt"  # separeted words file
    model_path = "information_model0830_pre_extra"  # model file

    source_separated_words_file = separate_word(filename, user_dic_file, separated_file)
    source_separated_words_file = separated_file  # if separated word file exist, don't separate_word again
    build_model(source_separated_words_file, model_path)  # if model file is exist, don't buile modl

    model = load_models(model_path)
    words = get_similar_words_str('医院', model)
    print(words)

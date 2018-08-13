from nltk.tokenize import WordPunctTokenizer
import io
import numpy as np

def wordtokenizer(sentence):
    # 分段
    words = WordPunctTokenizer().tokenize(sentence)
    return words

# path = '/unsullied/sharefs/ouyangzhihao/isilon-home/AAAI/mos/data/penn/train.txt'
# with io.open(path, encoding='utf-8') as f:
#     text = f.read().lower()

# print('text',text[1:100])
# words = wordtokenizer(text)
# print(np.shape(words))
# print(type(words))
# print(words[0:10])
# print('type of word:',type(words[1]))
# import ipdb; ipdb.set_trace()
# exit()
def get_text_train_data(time_step = 10, infor_length = 15, how_much_part = 1):

    path = '/unsullied/sharefs/ouyangzhihao/isilon-home/AAAI/mos/data/penn/train.txt'
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    text = text[:int(len(text) / how_much_part)]
    print('truncated corpus length:', len(text))
    words = wordtokenizer(text)

    # cut the text in semi-redundant sequences of time_step characters
    step = 1

    sentences = []
    next_words = []
    for i in range(0, len(words) - infor_length, step):
        sentences.append(''.join(words[i: i + infor_length]))
        next_words.append(words[i + infor_length])


    # sentences_time_step = []
    # next_chars_time_step = []
    # for i in range(0, len(sentences) - time_step, step):
    #     sentences_time_step.append((sentences[i: i + time_step]))
    #     next_chars_time_step.append(next_chars[i + time_step])

    return sentences,next_words


class vector_pair:
    def __init__(self, input, label):
        self.labels = {}
        self.input = input
        self.add_label(label)
    def add_label(self,new_label):
        if not(new_label in self.labels):
            self.labels[new_label] = 1
        else:
            self.labels[new_label] += 1
    def get_acc(self):
        acc = 0.
        total_times = 0.
        for var in self.labels:
            total_times += self.labels[var]
            acc = max(acc,self.labels[var])
        acc = acc / total_times
        return acc
    def get_total_times_in_dataset(self):
        total_times = 0
        for var in self.labels:
            total_times += self.labels[var]
        return total_times

def calculate_res(text_pairs):
    acc = 0
    res_count = 0.
    for vec in text_pairs:
        acc_text_p = text_pairs[vec].get_acc()
        count_text_p = text_pairs[vec].get_total_times_in_dataset()

        acc += acc_text_p * count_text_p
        res_count += count_text_p
    return acc / res_count , res_count



def run(length):
    train_data, train_label = get_text_train_data(infor_length=length)
    print('Build model...')

    text_pairs = {}
    for index, var in enumerate(train_data):
        if(var in text_pairs.keys()):
            text_pairs[var].add_label(train_label[index])
        else:
            text_pairs[var] = vector_pair(var, train_label[index])

    print('Finish init!~')
    try:
        acc, _ = calculate_res(text_pairs)
        print(acc, _)
    except Exception as e:
        print(e)

    max_acc_log = './words_max_acc_maxAlg.txt'
    # num / Acc
    print('%d \t %f' % (length, acc), file=open(max_acc_log, 'a'))
    del text_pairs
    del train_data
    del train_label
    import gc
    gc.collect()

for i in range(1,41):
    run(i)
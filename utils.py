from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import os
def build_embeedding_news(embedding_fn, dictionary, data_dir):
    
    # print("building embedding matrix for dict %d if need..." % len(dictionary))
    embedding_mat_fn = os.path.join(data_dir, embedding_fn.split("/")[-1] + "embedding_mat_%d.npy" % (len(dictionary)))
    # print(embedding_mat_fn)

    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    word_vectors = KeyedVectors.load_word2vec_format(embedding_fn, binary=True)
    embedding_dim = len(word_vectors.word_vec("one"))
    embedding_mat = np.zeros((len(dictionary) + 1, embedding_dim))

    for word,i in dictionary.items():
        if word in word_vectors.vocab:
            embedding_vec = word_vectors.word_vec(word)
        else:
            embedding_vec = np.random.uniform(-0.01, 0.01, embedding_dim).astype("float32")
        embedding_mat[i] = embedding_vec

    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat

def preprocessing(data,label_to_ix,BATCH_SIZE,word_to_ix):
    new_data = []
    for i, note, j in data:
        templabel = [0.0] * len(label_to_ix)
        for jj in j:
            if jj in label_to_ix.keys():
                templabel[label_to_ix[jj]] = 1.0
        templabel = np.array(templabel, dtype=float)
        new_data.append((i, note, templabel))
    new_data = np.array(new_data)

    lenlist = []
    for i in new_data:
        lenlist.append(len(i[0]))
    sortlen = sorted(range(len(lenlist)), key=lambda k: lenlist[k],reverse = False)
    new_data = new_data[sortlen]

    batch_data = []

    for start_ix in range(0, len(new_data) - BATCH_SIZE + 1, BATCH_SIZE):
        thisblock = new_data[start_ix:start_ix + BATCH_SIZE]
        mybsize = len(thisblock)
        numword = np.max([len(ii[0]) for ii in thisblock])
        main_matrix = np.zeros((mybsize, numword), dtype=np.int)
        for i in range(main_matrix.shape[0]):
            for j in range(len(thisblock[i][0])):
                if thisblock[i][0][j] in word_to_ix:
                    main_matrix[i, j] = word_to_ix[thisblock[i][0][j]]

        xxx2 = []
        yyy = []
        for ii in thisblock:
            xxx2.append(ii[1])
            yyy.append(ii[2])

        xxx2 = np.array(xxx2)
        yyy = np.array(yyy)
        batch_data.append((main_matrix,xxx2,yyy))
    return batch_data
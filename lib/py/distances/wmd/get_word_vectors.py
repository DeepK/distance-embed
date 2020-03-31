import numpy as np


# read datasets line by line
def read_line_by_line(dataset_name,C,model,vec_size):
    # get stop words (except for twitter!)
    SW = set()
    for line in open('wmd/stop_words.txt'):
        line = line.strip()
        if line != '':
            SW.add(line)

    stop = list(SW)


    f = open(dataset_name)
    if len(C) == 0:
        C = np.array([], dtype=np.object)
    num_lines = sum(1 for line in open(dataset_name))
    y = np.zeros((num_lines,))
    X = np.zeros((num_lines,), dtype=np.object)
    BOW_X = np.zeros((num_lines,), dtype=np.object)
    count = 0
    remain = np.zeros((num_lines,), dtype=np.object)
    the_words = np.zeros((num_lines,), dtype=np.object)
    for line in f:
        print ('%d out of %d' % (count+1, num_lines))
        line = line.strip()
        T = line.split('\t')
        classID = T[0]
        if classID in C:
            IXC = np.where(C==classID)
            y[count] = IXC[0]+1
        else:
            C = np.append(C,classID)
            y[count] = len(C)
        W = line.split()
        F = np.zeros((vec_size,len(W)-1))
        inner = 0
        RC = np.zeros((len(W)-1,), dtype=np.object)
        word_order = np.zeros((len(W)-1), dtype=np.object)
        bow_x = np.zeros((len(W)-1,))
        for word in W[1:len(W)]:
            try:
                test = model[word]
                if word in stop:
                    word_order[inner] = ''
                    continue
                if word in word_order:
                    IXW = np.where(word_order==word)
                    bow_x[IXW] += 1
                    word_order[inner] = ''
                else:
                    word_order[inner] = word
                    bow_x[inner] += 1
                    F[:,inner] = model[word]
            except:
                #print 'Key error: "%s"' % str(e)
                word_order[inner] = ''
            inner = inner + 1
        Fs = F.T[~np.all(F.T == 0, axis=1)]
        word_orders = word_order[word_order != '']
        bow_xs = bow_x[bow_x != 0]
        X[count] = Fs.T
        the_words[count] = word_orders
        BOW_X[count] = bow_xs
        count = count + 1;
    return (X,BOW_X,y,C,the_words)

import numpy as np

def parse(path_data, path_labels, shave):

    if shave:
        reduce_by = 2
    else:
        reduce_by = 1

    file1 = open(path_data, 'r')
    Lines = np.array(file1.readlines())
    file2 = open(path_labels, 'r')
    lines_labels = np.array(file2.readlines())

    chars = []
    labels = []
    lines_per_image = Lines.shape[0]//lines_labels.shape[0]
    chars_per_line = len(Lines[0])
    for index, line in enumerate(Lines):
        chars.append(np.array(list(line))[:-reduce_by])
        if index % lines_per_image == 0:
            labels.append(int(lines_labels[index//lines_per_image][:-1]))
    chars = np.array(chars)
    labels = np.array(labels)

    chars = np.reshape(chars, [Lines.shape[0]//lines_per_image, lines_per_image, chars_per_line - reduce_by])
    if shave:
        chars = chars[:,1:,:]
    images = (chars != " ").astype(float)


    return images, labels
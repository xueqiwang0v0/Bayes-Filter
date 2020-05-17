import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    #print(cmap)
    #plt.imshow(cmap)
    actions = data['arr_1']
    #print(actions)
    observations = data['arr_2']
    #print(observations)
    belief_states = data['arr_3']

    #### Test your code here
    test = HistogramFilter()
    belief = np.ones((20,20))
    belief[belief==1] = 0.0025
    for i in range(np.shape(actions)[0]):
    #for i in range(1):
        print(i)
        ac = actions[i]
        print('action',ac)
        ob = observations[i]
        #print(cmap.shape,belief.shape)
        first = test.histogram_filter(cmap,belief,ac,ob)
        belief = first[0]
        pos = first[1]
        print(belief_states[i],np.array(pos))
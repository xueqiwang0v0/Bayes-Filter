import numpy as np
import matplotlib.pyplot as plt

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''
        x_max = cmap.shape[1]
        y_max = cmap.shape[0]
        
        #make cmap[x,y]
        cmap = np.flipud(cmap)
        cmap = np.transpose(cmap)
        
        p_color = np.where(cmap == observation, 0.9, 0.1) # the color possibilities
        
        movex = action[0]
        movey = action[1]
        
        #if the robot doesn't move
        post_belief = belief * p_color * 0.1

        #if the robot moves        
        if movex == 1:
            post_belief[1:x_max,:] += belief[0:x_max-1,:] * p_color[1:x_max,:] * 0.9
            post_belief[-1,:] += belief[-1,:] * p_color[-1,:] * 0.9  
        
        elif movex == -1:
            post_belief[0:x_max-1,:] += belief[1:x_max,:] * p_color[0:x_max-1,:] * 0.9
            post_belief[0,:] += belief[0,:] * p_color[0,:] * 0.9  
        elif movey == 1:
            post_belief[:,1:y_max] += belief[:,0:y_max-1] * p_color[:,1:y_max] * 0.9
            post_belief[:,-1] += belief[:,-1] * p_color[:,-1] * 0.9
        elif movey == -1:
            post_belief[:,0:y_max-1] += belief[:,1:y_max] * p_color[:,0:y_max-1] * 0.9
            post_belief[:,0] += belief[:,0] * p_color[:,0] * 0.9
        
        #find the most possible robot position
        pos = np.unravel_index(np.argmax(post_belief),post_belief.shape)
        pos = np.array(pos)
        
        return [post_belief, pos]

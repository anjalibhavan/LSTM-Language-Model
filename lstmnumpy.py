import numpy as np
learning_rate = 0.01
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def softmax(y):
    return np.exp(y) / np.sum(np.exp(y))

def activation_fn(res, activation):
    if(activation=='relu'):
        return np.max(0,res)
    elif(activation=='softmax'):
        return softmax(res)
    elif(activation=='sigmoid'):
        return sigmoid(res)
    elif(activation=='tanh'):
        return tanh(res)
    

class lstmnumpy:
    def __init__(self, hidden_size, embedding_size, bptt, activation, **keyword_parameters):
        
        super(lstmnumpy, self).__init__()
        self.conc_size = hidden_size + embedding_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.W_i = np.random.randn(hidden_size,self.conc_size)
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_f = np.random.randn(hidden_size,self.conc_size)
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size,self.conc_size)
        self.b_o = np.zeros((hidden_size, 1))
         
        self.W_c = np.random.randn(hidden_size,self.conc_size)
        self.b_c = np.zeros((hidden_size, 1))
        
        self.W_y = np.random.randn(embedding_size,hidden_size)
        self.b_y = np.zeros((embedding_size, 1))
        
        self.h_prev = np.zeros((bptt, hidden_size, 1))
        self.c_prev = np.zeros((bptt, hidden_size, 1))

        self.dW_f = np.zeros_like(self.W_f)
        self.dW_i = np.zeros_like(self.W_i)
        self.dW_c = np.zeros_like(self.W_c)

        self.dW_o = np.zeros_like(self.W_o)
        self.dW_y = np.zeros_like(self.W_y)

        self.db_f = np.zeros_like(self.b_f)
        self.db_i = np.zeros_like(self.b_i)
        self.db_c = np.zeros_like(self.b_c)

        self.db_o = np.zeros_like(self.b_o)
        self.db_y = np.zeros_like(self.b_y)
        
        self.mW_f = np.zeros_like(self.W_f)
        self.mW_i = np.zeros_like(self.W_i)
        self.mW_c = np.zeros_like(self.W_c)
        self.mW_o = np.zeros_like(self.W_o)
        self.mW_y = np.zeros_like(self.W_y)
        
        self.mb_f = np.zeros_like(self.b_f)
        self.mb_i = np.zeros_like(self.b_i)
        self.mb_c = np.zeros_like(self.b_c)
        self.mb_o = np.zeros_like(self.b_o)
        self.mb_y = np.zeros_like(self.b_y)
        
        self.activation = activation
        
        self.lastlayer = False
        
        if('last' in keyword_parameters):
            self.lastlayer = keyword_parameters['last']
        
    def forward(self , x, **keyword_parameters):
        
        if ('h_prev' in keyword_parameters):
            h_prev = keyword_parameters['h_prev']
        if('c_prev' in keyword_parameters):
            c_prev = keyword_parameters['c_prev']
        else:
            h_prev = self.h_prev
            c_prev = self.c_prev
            
        assert x.shape == (self.embedding_size, 1)
        assert h_prev.shape == (self.bptt, self.hidden_size, 1)
        assert c_prev.shape == (self.bptt, self.hidden_size, 1)
        
        z = np.row_stack((h_prev,x))
        
        i = sigmoid(np.dot(self.W_i,z) + self.b_i)
        f = sigmoid(np.dot(self.W_f,z) + self.b_f)
        o = sigmoid(np.dot(self.W_o,z) + self.b_o)
        c_bar = tanh(np.dot(self.W_c,z)+self.b_c)
        
        c = f * c_prev + i * c_bar
        
        h = o * tanh(c)
        
        y = np.dot(self.W_y, h) + self.b_y
        p = softmax(y)
    
        parameters = {}
        
        parameters['z'] = z
        parameters['f'] = f
        parameters['i'] = i
        parameters['c_bar'] = c_bar
        parameters['c'] = c
        parameters['o'] = o
        parameters['h'] = h
        parameters['y'] = y
        parameters['p'] = p
        parameters['c_prev'] = c_prev
        
        if(self.lastlayer == True):
            return np.argmax(p,axis = 1), parameters
        else:
            return activation_fn(h,self.activation), parameters

    
    def backward(self, target, dh_next, dc_next, parameters):

    
        assert parameters['z'].shape == self.conc_size
        assert parameters['y'].shape == self.embedding_size
        assert parameters['p'].shape == self.embedding_size
    
        for param in [dh_next, dc_next, parameters['c_prev'], parameters['f'], parameters['i'], 
                      parameters['c_bar'], parameters['c'], parameters['o'], parameters['h']]:
            assert param.shape == (self.hidden_size, 1)
    
        dy = np.copy(parameters['p'])
        dy[target] -= 1
    
        self.dW_y += np.dot(dy, parameters['h'].T)
        self.db_y += dy
    
        dh = np.dot(self.W_y.T, dy)
        dh += dh_next
        do = dh * tanh(parameters['c'])
        do = dsigmoid(parameters['o']) * do
        self.dW_o += np.dot(do, parameters['z'].T)
        self.db_o += do
    
        dc = np.copy(dc_next)
        dc += dh * parameters['o'] * dtanh(tanh(parameters['c']))
        dc_bar = dc * parameters['i']
        dc_bar = dc_bar * dtanh(parameters['c_bar'])
        self.dW_c += np.dot(dc_bar, parameters['z'].T)
        self.db_c += dc_bar
    
        di = dc * parameters['c_bar']
        di = dsigmoid(parameters['i']) * di
        self.dW_i += np.dot(di, parameters['z'].T)
        self.db_i += di
    
        df = dc * parameters['c_prev']
        df = dsigmoid(parameters['f']) * df
        self.dW_f += np.dot(df, parameters['z'].T)
        self.db_f += df
    
        dz = np.dot(self.W_f.T, df) \
            + np.dot(self.W_i.T, di) \
            + np.dot(self.W_c.T, dc_bar) \
            + np.dot(self.W_o.T, do)
        dh_prev = dz[:self.hidden_size, :]
        dc_prev = parameters['f'] * dc
    
        return dh_prev, dc_prev
    
    def forward_sequence(self,inputs, **keyword_parameters):
    # To store the values for each time step
        x_s, z_s, f_s, i_s, c_bar_s, c_s, o_s, h_s, y_s, p_s = np.zeros((self.embedding_size, inputs.shape[1])), {}, {}, {}, {}, np.zeros((self.hidden_size, inputs.shape[1])), {}, np.zeros((64, inputs.shape[1])), {}, {}
        
        if ('h_prev' in keyword_parameters):
            h_prev = keyword_parameters['h_prev']
        if('c_prev' in keyword_parameters):
            c_prev = keyword_parameters['c_prev']
        else:
            h_prev = self.h_prev
            c_prev = self.c_prev
            
        # Values at t - 1
        h_s[:,-1:] = np.copy(h_prev)
        c_s[:,-1:] = np.copy(c_prev)
    
        #loss = 0
        # Loop through time steps
        #BigH = np.zeros([1, inputs.shape[1], inputs.shape[2]])
        for t in range(40):
            x_s[:, t] = inputs[:, t]  # Input character
            params={}
            unreqh, params = self.forward(np.reshape(x_s[:, t], (self.embedding_size,1)), 
                                          h_prev = np.reshape(h_s[:, t - 1], (self.hidden_size,1)), 
                                          c_prev = np.reshape(c_s[:, t - 1], (self.hidden_size,1))) # Forward pass
            
            z_s[t], f_s[t], i_s[t] = params['z'], params['f'], params['i']
            c_bar_s[t], c_s[:, t], o_s[t] = params['c_bar'], params['c'], params['o']
            h_s[:,t], y_s[t], p_s[t] = params['h'], params['y'], params['p']
    
            #loss += -np.log(p_s[t][targets[t], 0]) # Loss for at t
            
        cache = {}
        cache['z'] = z_s
        cache['f'] = f_s
        cache['i'] = i_s
        cache['c_bar'] = c_bar_s
        cache['c'] = c_s
        cache['o'] = o_s
        cache['h'] = h_s
        cache['y'] = y_s
        cache['p'] = p_s
            
        return h_s, cache
        
    
    def backward_sequence(self,inputs, targets, cache):
        for dparam in [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y, self.db_f, self.db_i, self.db_c, self.db_o, self.db_y]:
            dparam.fill(0)
    
        dh_nexts = np.zeros_like(cache['h'][0]) #dh from the next character
        dc_nexts = np.zeros_like(cache['c'][0]) #dh from the next character
    
        for t in reversed(range(40)):
            # Backward pass
            newparams = {}
            newparams['z'] = cache['z'][t]
            newparams['f'] = cache['f'][t]
            newparams['i'] = cache['i'][t]
            newparams['c_bar'] = cache['c_bar'][t]
            newparams['c'] = cache['c'][t]
            newparams['o'] = cache['o'][t]
            newparams['h'] = cache['h'][t]
            newparams['y'] = cache['y'][t]
            newparams['p'] = cache['p'][t]
            dh_next, dc_next = self.backward(newparams, target = targets[t], dh_next = dh_nexts, dc_next = dc_nexts)
    
        # clip gradients to mitigate exploding gradients
        for dparam in [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y, self.db_f, self.db_i, self.db_c, self.db_o, self.db_y]:
            np.clip(dparam, -1, 1, out=dparam)
    
        return newparams['h'][inputs.shape[2] - 1], newparams['c'][inputs.shape[2] - 1]
    
    
    def update_params(self):
        
        for param, dparam, mem in zip([self.W_f, self.W_i, self.W_c, self.W_o, self.W_y, self.b_f, self.b_i, self.b_C, self.b_o, self.b_y],
                                      [self.dW_f, self.dW_i, self.dW_c, self.dW_o, self.dW_y, self.db_f, self.db_i, self.db_c, self.db_o, self.db_y],
                                      [self.mW_f, self.mW_i, self.mW_c, self.mW_o, self.mW_y, self.mb_f, self.mb_i, self.mb_c, self.mb_o, self.mb_y]):
            mem += dparam * dparam # Calculate sum of gradients
            #print(learning_rate * dparam)
            param += -(learning_rate * dparam / np.sqrt(mem + 1e-8))
    
    
    #def predict(self, x, **keyword_parameters):
     #   h, p = self.forward_propagation(x)
      ##     return np.argmax(p, axis=1)
        #else:
         #   return h
        
        
        
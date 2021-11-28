# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:50:29 2018
data formulation: (s[i], a[i], r[i])
@author: mengxiaomao
"""
from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r,s_next):
        for i in range(len(s)):
            experience = (s[i], a[i], r[i],s_next[i])
            if self.count < self.buffer_size: 
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)
            
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        minibatch = []
        if self.count < batch_size:
            minibatch = random.sample(self.buffer, self.count)
        else:
            minibatch = random.sample(self.buffer, batch_size)

        batch_s = [d[0] for d in minibatch]
        batch_a = [d[1] for d in minibatch]
        batch_r = [d[2] for d in minibatch]
        batch_s_next = [d[3] for d in minibatch]
        return batch_s, batch_a, batch_r,batch_s_next

    def clear(self):
        self.buffer.clear()
        self.count = 0




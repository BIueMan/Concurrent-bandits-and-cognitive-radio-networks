import numpy as np

class CircularAveragingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.int8)
        self.index = 0
        self.count = 0
        self.sum = np.int32(0)  # Avoid overflow from int8 summing

    def add(self, value):
        value = np.int8(value)
        old_value = self.buffer[self.index]
        self.sum -= np.int32(old_value)
        self.buffer[self.index] = value
        self.sum += np.int32(value)

        self.index = (self.index + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def average(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def get_elements(self):
        if self.count < self.size:
            return self.buffer[:self.count]
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))
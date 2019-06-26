class Layer(object):

    def __init__(self, NodeCount, Next, Activation):
        self.NodeCount = NodeCount
        self.Next = Next;
        self.Activate = Activation

class InputLayer(Layer):

    def __init__(self, NodeCount, Next):
        Layer.__init__(self, NodeCount, Next, None)


class HiddenLayer(Layer):

    def __init__(self, NodeCount, Next, Activation):
        Layer.__init__(self, NodeCount, Next, Activation)


class OutputLayer(Layer):

    def __init__(self, NodeCount, Activation):
        Layer.__init__(self, NodeCount, None, Activation)

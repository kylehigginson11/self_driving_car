import unittest
from unittest.mock import patch

import sys
sys.path.append("../")
import os

from neural_network_training.train_mlp import TrainMLP


class TestNeuralNetwork(unittest.TestCase):

    @patch('neural_network_training.train_mlp.TrainMLP', 'create_mlp', return_value="PASS")
    def test_creation(self):
        ann_name = "test"
        ann_location = 'neural_network/{}_neural_network.xml'.format(ann_name)
        trainer = TrainMLP(ann_name)


if __name__ == '__main__':
    unittest.main()
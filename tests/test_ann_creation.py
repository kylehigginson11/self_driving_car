from unittest import TestCase
from unittest.mock import patch
from neural_network_training.collect_training_images import CollectTrainingImages


class TestNeuralNetwork(TestCase):

    @patch('main.Blog')
    def test_sum(self, sum):
        self.assertEqual(sum(2,3), 9)
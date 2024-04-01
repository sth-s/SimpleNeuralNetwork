import numpy as np
from NeuralNetwork import NeuralNetwork
from PIL import Image
import os

class NumbersModelTrainer:
    def __init__(self, nn: NeuralNetwork, train_directory: str, samples: int):
        self.nn = nn
        self.train_directory = train_directory
        self.samples = samples

    def load_data(self):
        images = []
        digits = []
        images_files = [f for f in os.listdir(self.train_directory) if os.path.isfile(os.path.join(self.train_directory, f))]
        for i, file_name in enumerate(images_files[:self.samples]):
            img = Image.open(os.path.join(self.train_directory, file_name)).convert('L')
            images.append(np.asarray(img))
            digits.append(int(file_name[10]))

        inputs = np.zeros((self.samples, 784))
        for i, img in enumerate(images):
            for x in range(28):
                for y in range(28):
                    inputs[i, x + y * 28] = img[y, x] / 255.0

        return inputs, digits

    def train(self, epochs: int):
        inputs, digits = self.load_data()
        steps = 0
        for epoch in range(epochs):
            right = 0
            error_sum = 0
            for index in range(self.samples):
                targets = np.zeros(10)
                digit = digits[index]
                targets[digit] = 1

                outputs = self.nn.feed_forward(inputs[index])
                max_digit = np.argmax(outputs)
                if digit == max_digit:
                    right += 1
                error_sum += np.sum((targets - outputs) ** 2)

                self.nn.backpropagation(inputs[index], targets)

                steps += 1 
                #print(f"step: {steps}")
            print(f"epoch: {epoch + 1}. correct: {right/self.samples}. error: {error_sum/self.samples}")
            save_path = "./trained_model_config_backup.json"
            NeuralNetwork.save_nn_config(self.nn, save_path)
            

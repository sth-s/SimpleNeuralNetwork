from NeuralNetwork import NeuralNetwork
from NumbersModelTrainer import NumbersModelTrainer
from Painter import Painter
import numpy as np
from PIL import Image
import os

def main():
    try:

        #                                                    # Creating a neural network
        # learning_rate = 0.001
        # activation = "sigmoid"
        # derivative = "sigmoid_derivative"
        # sizes = [784, 512, 128, 32, 10]
        # nn = NeuralNetwork(learning_rate, activation, derivative, *sizes)

        # # Path to training data directory
        # train_directory = "./train"

        # # Create a trainer instance and start the training process
        # trainer = NumbersModelTrainer(nn, train_directory, samples=60000)
        # trainer.train(epochs=100)

        # # Preserving the configuration of the trained model
        # save_path = "./trained_model_config.json"
        # NeuralNetwork.save_nn_config(nn, save_path)

        
        #                                                    # Loading the trained model configuration
        # # Specify the path to the neural network configuration file
        # load_path = "./trained_model_config.json"

        # # Load the configuration of the neural network
        # nn_loaded = NeuralNetwork.load_nn_config(load_path)


        #                                                    # Testing the trained model
        # samples = 10000

        # def load_data():
        #     images = []
        #     digits = []
        #     images_files = [f for f in os.listdir(train_directory) if os.path.isfile(os.path.join(train_directory, f))]
        #     for i, file_name in enumerate(images_files[:samples]):
        #         img = Image.open(os.path.join(train_directory, file_name)).convert('L')
        #         images.append(np.asarray(img))
        #         digits.append(int(file_name[10]))

        #     inputs = np.zeros((samples, 784))
        #     for i, img in enumerate(images):
        #         for x in range(28):
        #             for y in range(28):
        #                 inputs[i, x + y * 28] = img[y, x] / 255.0

        #     return inputs, digits
        
        # inputs, digits = load_data()

        # right = 0
        # error_sum = 0

        # for index in range(samples):
        #     targets = np.zeros(10)
        #     digit = digits[index]
        #     targets[digit] = 1

        #     outputs = nn_loaded.feed_forward(inputs[index])
        #     max_digit = np.argmax(outputs)
        #     if digit == max_digit:
        #         right += 1
        #     error_sum += np.sum((targets - outputs) ** 2)

        # print(f"correct: {right/samples}. error: {error_sum/samples}")



        #                                                   # Running the application
        # app = Painter(nn_loaded)
        # app.mainloop()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

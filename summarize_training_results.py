import matplotlib.pyplot as plt
import pickle
import os.path


def plot_training_statistics(history):
    """ Plots the fit history statistics like training and validation loss. 
    
    history -- History of model training ['loss', 'val_loss'].
    """
    plt.plot(history['loss'], 'x-')
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.grid()
    print('Close the figures to continue...')
    plt.show()


def main():
    """ Plot training results. """

    # unpickle history object
    filename = './history.obj'

    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            history = pickle.load(file)
            plot_training_statistics(history)
    else:
        print('History object file \"{:s}\" not found.'.format(filename))
        exit(-1)

if __name__ == "__main__":
    main()

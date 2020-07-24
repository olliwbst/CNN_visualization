import keras
from vis.utils import utils
from vis.input_modifiers import Jitter
from vis.visualization import visualize_cam, visualize_activation, visualize_saliency, get_num_filters
import innvestigate.utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import random
import os

'''settable parameters start'''
'''paths'''
model_PATH = '/content/drive/My Drive/cnn_models/graydog.h5'

X_train_PATH = '/content/drive/My Drive/cnn_ready_data/graydog_features_50_1.npy'  # train-features
y_train_PATH = '/content/drive/My Drive/cnn_ready_data/graydog_label_50_1.npy'  # train-labels

X_test_PATH = '/content/drive/My Drive/cnn_ready_data/graydog_test_features_50_1.npy'  # test-features
y_test_PATH = '/content/drive/My Drive/cnn_ready_data/graydog_test_label_50_1.npy'  # test-labels

IMG_DIR = r'/content/result_images'  # img-directory for saving results

'''lower this in case you run into problems (L.75-78)'''
analyzer_batchsize = 32

'''dont change the following unless you have a custom cnn-structure'''
name_softmax_layer = 'predictions'  # used for vis-functions
name_last_conv_layer = 'block2_conv2'  # used for gradcam-function
layerlist_for_layered_actmax = ['block1_conv1', 'block1_conv2', 'block1_pool',
                                'block2_conv1', 'block2_conv2', 'block2_pool',
                                'fc1', 'fc2']  # used for layered actmax-function
'''settable parameters end'''

'''loading and reshaping data'''
print('loading data...')
model = keras.models.load_model(model_PATH)
linear_model = keras.models.load_model(model_PATH)
linear_model.layers[utils.find_layer_idx(linear_model, 'predictions')].activation = keras.activations.linear
linear_model = utils.apply_modifications(linear_model)

X_train = np.load(X_train_PATH)
X_train = X_train / 255
y_train = np.load(y_train_PATH)

X_test = np.load(X_test_PATH)
X_test = X_test / 255
y_test = np.load(y_test_PATH)
print('loaded data...')

y_test = keras.utils.to_categorical(y_test)  # reshapes labels to categorical format
IMG_SIZE = X_test.shape[1]  # sets to image dimensions for plotting equal to those of used data
IMG_DEPTH = X_test.shape[3]  # getting information whether input data is grayscale (1) or color (3)

'''creating result directory if it doesnt already exist by the set name'''
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
    print("Directory ", IMG_DIR,  " created!")
else:
    print("Directory ", IMG_DIR,  " already exists, skipping.")


'''stripping softmax off of the model, needed for innvestigate functions'''
def strip_softmax(_model):
    _model = keras.models.Model(inputs=_model.get_input_at(0), outputs=_model.layers[-2].get_output_at(0))

    return _model


'''calculation analyzers for patternnet and patternattribution-functions'''
print('calculating analysers...')
pn_analyzer = innvestigate.create_analyzer('pattern.net', strip_softmax(model), **{"pattern_type": "relu"})
pn_analyzer.fit(X_train, batch_size=analyzer_batchsize, verbose=1)
pa_analyzer = innvestigate.create_analyzer('pattern.attribution', strip_softmax(model), **{"pattern_type": "relu"})
pa_analyzer.fit(X_train, batch_size=analyzer_batchsize, verbose=1)
print('calculated analysers...')

'''checks compatibility of model and inputdatasets, adjusts them is incompatible
should be run when you introduce different sets or run into numpy related 
compatibility-errors, this might solve them'''
def reshape_testset():
    print('checking compatibility of loaded testset and model...')
    modelinputshape = model.layers[0].input_shape
    reshaped = []
    global X_test

    if modelinputshape[1:] == X_test.shape[1:]:
        print('testset and model compatible, no need to reshape!')
    else:
        print('testset and model incompatible, reshaping testset to match model...')
        for i in X_test:
            i = i.reshape(X_test.shape[1], X_test.shape[2])
            i = cv2.resize(i, (modelinputshape[1], modelinputshape[2]))
            reshaped.append(i)
        reshaped = np.array(reshaped).reshape(-1, modelinputshape[1], modelinputshape[2], modelinputshape[3])
        X_test = reshaped
        print('reshaped testset!')


'''prints an evaluation of the models performance on the loaded testset'''
def print_evaluation():
    eval = model.evaluate(X_test, y_test)

    print('loss: {}, accuracy: {}'.format(eval[0], eval[1]))


'''predicts a sample by its index in the testset, 0 would give the prediction
for the first one etc.'''
def predict_sample(sample_id):
    img_array = np.array(X_test[sample_id]).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    prediction = model.predict(img_array)
    print('Model-Predictions: {},Index-Labels: {}'.format(prediction, y_test[sample_id]))


'''predicts the classifications for a given range of indices (takes a list),
if no range is given, it predicts the whole testset'''
def predict_range(_range=None):
    if _range is None:
        _range = len(X_test)

    predictions = model.predict(X_test[:_range])

    return predictions


'''returns a list of indices from X_test whose predictions by the model were wrong'''
def get_false_predictions(_range=None):
    if _range is None:
        _range = len(X_test)
    negatives = []
    j = 0

    print('predicting {} samples...'.format(_range))
    predictions = model.predict(X_test[:_range])
    print('got predictions! comparing...')

    for i in predictions:
        for classes in range(len(i)):
            x = round(i[classes])
            y = y_test[j, classes]
            if x != y:
                if j in negatives:
                    continue
                else:
                    negatives.append(j)
        j += 1

    print('Number of false predictions: {} of {} tested.'.format(len(negatives), _range))

    return negatives


'''returns the average prediction confidence of given indices from the testset'''
def get_average_confidence(indices):
    average_confidence = 0.0
    counter = 0
    for index in indices:
        img_array = np.array(X_test[index]).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
        prediction = model.predict(img_array)
        average_confidence += max(max(prediction))
        counter += 1
    average_confidence /= counter

    return average_confidence


'''prints a summary of the architecture of the currently loaded model'''
def print_summary():
    model.summary()


'''plots the desired samples from testset by indices (takes a list) with correct
and model predictions as the plot-title'''
def plot_index_sample(indices):
    for index in indices:
      img_array = np.array(X_test[index]).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
      img = img_array * 255
      img = img.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
      img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
      prediction = model.predict(img_array)

      for i in range(len(y_test[index])):
            if y_test[index, i] == 1:
                correct_pred = i

      fig = plt.figure('Index: {}, correct prediction: class{}, model-predictions: {}'.format(index, correct_pred ,prediction))
      # the print is here for google colab, as it cant show the title there
      print('Index: {}, correct prediction: class{}, model-predictions: {}'.format(index, correct_pred ,prediction))
      plt.imshow(img)
      plt.show()


'''deep taylor decomposition-function'''
def dtd(sample):
    # Create analyzer
    analyzer = innvestigate.create_analyzer('deep_taylor', strip_softmax(model))

    # sample in format expected by model
    x = sample

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    a = a.reshape(IMG_SIZE, IMG_SIZE)

    return a


'''guided backprop-function'''
def guided_backprop(sample):
    # Create analyzer
    analyzer = innvestigate.create_analyzer('guided_backprop', strip_softmax(model))

    # sample in format expected by model
    x = sample

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    a = a.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return a


'''patternnet-function'''
def patternnet(sample):
    # sample in format expected by model
    x = sample

    # Apply analyzer w.r.t. maximum activated output-neuron
    # a = analyser.analyse(x)
    a = pn_analyzer.analyze(x)

    a = a.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return a


'''pattern attribution-function'''
def patternattribution(sample):
    # sample in format expected by model
    x = sample

    # Apply analyzer w.r.t. maximum activated output-neuron
    # a = analyser.analyse(x)
    a = pa_analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    a = a.reshape(IMG_SIZE, IMG_SIZE)

    return a


'''activation maximization-function'''
def actmax(sample):
    cams = []
    for i in range(len(y_test[0])):
        cam = visualize_activation(linear_model, utils.find_layer_idx(model, name_softmax_layer), filter_indices=i,
                                   wrt_tensor=None, seed_input=sample, input_range=(0, 255), backprop_modifier=None,
                                   grad_modifier=None, act_max_weight=1, lp_norm_weight=10, tv_weight=10)

        cam = cam.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
        cam = cv2.cvtColor(cam.astype('uint8'), cv2.COLOR_BGR2RGB)
        cams.append(cam)

    return cams


'''saliency-maps-function'''
def saliency(sample):
    cams = []
    for i in range(len(y_test[0])):
        cam = visualize_saliency(linear_model, utils.find_layer_idx(linear_model, name_softmax_layer), filter_indices=i,
                                 seed_input=sample, wrt_tensor=None, backprop_modifier='guided', grad_modifier='relu',
                                 keepdims=False)
        cams.append(cam)

    return cams


'''gradcam-function'''
def gradcam(sample):
    cams = []
    for i in range(len(y_test[0])):
        cam = visualize_cam(linear_model, utils.find_layer_idx(linear_model, name_softmax_layer), filter_indices=i,
                            seed_input=sample, penultimate_layer_idx=utils.find_layer_idx(linear_model, name_last_conv_layer),
                            backprop_modifier='guided', grad_modifier=None)
        cams.append(cam)

    return cams


'''gets sample_count number of filters (randomly) out of each given layer in the 
global inputs at the start of the script and saves them to the IMG_DIR'''
def layered_actmax(sample_count):
    for layer_nm in layerlist_for_layered_actmax:
        layer_idx = utils.find_layer_idx(model, layer_nm)
        num_filters = get_num_filters(model.layers[layer_idx])
        drawn_filters = random.choices(np.arange(num_filters), k=sample_count)
        for filter_id in drawn_filters:
            img = visualize_activation(model, layer_idx, filter_indices=filter_id, input_modifiers=[Jitter(16)])
            img = img.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, cmap='gray')
            img_path = os.path.join(IMG_DIR, layer_nm + '_' + str(filter_id) + '.jpg')
            plt.imsave(img_path, img)
            print(f'Saved layer {layer_nm}/{filter_id} to file!')
    print('done!')


'''gets all the finished visualizations for vis, plots them and saves the 
plotted image to IMG_DIR'''
def plot_vis(grads, actmaxs, sals, sample, correct_pred, sample_id='undefined'):
    prediction = model.predict(sample)  # get model prediction for sample
    '''prep sample'''
    img = sample * 255
    img = img.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)

    '''plot all vis-elements on same figure'''
    fig, axes = plt.subplots(3, (len(grads)+1), figsize=(3+(4*len(grads)), 10))
    # plt.subplots_adjust(wspace=0.25)  # margin between subplots
    '''gradcam'''
    axes[0, 0].imshow(img)  # what is shown in the subplot?
    axes[0, 0].title.set_text('Input Image')  # title of the subplot
    for grad in range(len(grads)):
        axes[0, grad + 1].imshow(img)
        i = axes[0, grad + 1].imshow(grads[grad], cmap='jet', alpha=0.9)  # generates an overlay over subplot at index 1
        axes[0, grad + 1].title.set_text('gradCAM, Class{}'.format(grad))
        divider = make_axes_locatable(
            axes[0, grad + 1])  # helper to adjust the sizing of plot and corresponding colorbar
        cax = divider.append_axes('right', size='5%', pad=0.05)  # helper to adjust the sizing of plot and corresp. cb
        fig.colorbar(i, cax=cax)  # displays colorbar for overlay 'i'
    '''actmax'''
    axes[1, 0].imshow(img)
    axes[1, 0].title.set_text('Input Image')
    for act in range(len(actmaxs)):
        axes[1, act + 1].imshow(actmaxs[act])
        axes[1, act + 1].title.set_text('Activation Maximization, Class{}'.format(act))
    '''saliency'''
    axes[2, 0].imshow(img)
    axes[2, 0].title.set_text('Input Image')
    for sal in range(len(sals)):
        axes[2, sal + 1].imshow(sals[sal], cmap='seismic', clim=(-1, 1))
        axes[2, sal + 1].title.set_text('Saliency Map, Class{}'.format(sal))

    labelstring = ''
    for i in range(len(prediction[0])):
        labelstring += ' Class {}: '.format(i)
        labelstring += '{:5.3f}'.format(prediction[0, i])

    plt.suptitle('Results for vis-library, Correct Class: {}, Model-Predictions:{}'.format(correct_pred, labelstring))

    img_path = os.path.join(IMG_DIR, '{}_vis.png'.format(sample_id))

    plt.savefig(img_path)


'''gets all the finished visualizations for innvestigate, plots them and saves the 
plotted image to IMG_DIR'''
def plot_innvestigate(gb, pn, pa, dtd, sample, correct_pred, sample_id='undefined'):
    prediction = model.predict(sample)
    '''prep sample'''
    img = sample * 255
    img = img.reshape(IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
    gb = cv2.cvtColor(gb, cv2.COLOR_BGR2RGB)
    pn = cv2.cvtColor(pn, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.25)
    axes[0].imshow(img)
    axes[0].title.set_text('Input Image')
    if IMG_DEPTH == 1:
        axes[1].imshow(gb)
        axes[2].imshow(pn)
    else:
        axes[1].imshow(gb)
        axes[2].imshow(pn)
    axes[1].title.set_text('GuidedBackprop')
    axes[2].title.set_text('PatternNet')
    axes[3].imshow(pa, cmap='seismic', clim=(-1, 1))
    axes[3].title.set_text('PatternAttribution')
    axes[4].imshow(dtd, cmap='seismic', clim=(-1, 1))
    axes[4].title.set_text('DeepTaylor')

    labelstring = ''
    for i in range(len(prediction[0])):
        labelstring += ' Class {}: '.format(i)
        labelstring += '{:5.3f}'.format(prediction[0, i])

    plt.suptitle('Results for Innvestigate-library, Correct Class: {}, Model-Predictions:{}'.format(correct_pred,
                                                                                                    labelstring))

    img_path = os.path.join(IMG_DIR, '{}_innvestigate.png'.format(sample_id))

    plt.savefig(img_path)


'''takes a list of indices from the testset, hands all necessary information to
all the visualization-implementations, gathers the results and hands them to
the plotting functions'''
def plot_index(samples):
    for sample in samples:
        img_array = np.array(X_test[sample]).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
        for i in range(len(y_test[sample])):
            if y_test[sample, i] == 1:
                correct_pred = i

        print('gathering and plotting visualizations for index {} (vis)...'.format(sample))
        plot_vis(gradcam(img_array), actmax(img_array), saliency(img_array), img_array, correct_pred, sample)

        print('gathering and plotting visualizations for index {} (innvestigate)...'.format(sample))
        plot_innvestigate(guided_backprop(img_array), patternnet(img_array),
                          patternattribution(img_array), dtd(img_array), img_array, correct_pred, sample)


'''used for displaying the gradients of a specific external image by its path,
handles data just like plot_index() also takes a name that the image is saved 
as, since there is no ground truth, only the models prediction, it wil hand 
'no ground-truth available' as ground truth information to the plotting functions'''
def plot_image(path, name='external_image'):
    if IMG_DEPTH == 1:
        img_array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # convert to gs array
    if IMG_DEPTH == 3:
        img_array = cv2.imread(str(path))  # convert to array

    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    img_array = img_array / 255

    correct_pred = 'no ground-truth available'

    print('gathering and plotting visualizations for your image (vis)...')
    plot_vis(gradcam(img_array), actmax(img_array), saliency(img_array), img_array, correct_pred, str(name))

    print('gathering and plotting visualizations for your image (innvestigate)...')
    plot_innvestigate(guided_backprop(img_array), patternnet(img_array),
                      patternattribution(img_array), dtd(img_array), img_array, correct_pred, str(name))

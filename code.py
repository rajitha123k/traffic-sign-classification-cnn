import pickle
import numpy as np
import tensorflow as tf
import random
import csv
import os
from PIL import Image
from sklearn.utils import shuffle
from keras.layers import Flatten
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
training_file = 'train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
validation_file = 'valid.p'
testing_file = 'test.p'


with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print(len(y_train))
print(len(y_valid))
print(len(y_test))

print(X_test.shape)
print(test['labels'])
# Number of training examples
n_train = len(y_train)

# Number of testing examples.
n_test = len(y_test)

# shape of an traffic sign image
image_shape = X_train.shape[1:4]

# unique classes/labels
n_classes = np.max(y_train) - np.min(y_train) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
def show_dataset_summary(pickle_dict):
    X, y = pickle_dict['features'], pickle_dict['labels']
    n_classes = np.max(y_train) - np.min(y_train) + 1
    n_data_of_classes = np.zeros((n_classes,))
    for i in range(n_classes):
        n_data_of_classes[i] = len(y[y == i])
        
    classes_num = [i for i in range(n_classes)]
    
    plt.figure()
    plt.bar(classes_num, n_data_of_classes, align="center", alpha=.5 )
    plt.xlabel('classes')
    plt.ylabel('#')
    plt.xlim([0-.5, classes_num[-1]+.5])
    plt.show()
def plot_test_images(images, nc = 15, nr = 4):
    ct = 0
    fig = plt.figure(figsize=(nc, nr))
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.0, hspace=0.0)
    for i in range(nr * nc):
        ax = fig.add_subplot(gs[ct])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(images[ct,:,:,:])
        ct += 1

    return fig
ind_ = np.random.randint(n_train, size=33)
fig = plot_test_images(X_train[ind_, :, :, :], nc=11, nr=3)
from PIL import Image
from PIL import ImageEnhance
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()
generate_distorted_images = False
if generate_distorted_images == True:
    N_rand_total = np.sum(np.floor(3000/n_data_of_classes) * n_data_of_classes)
    N_rand_total = N_rand_total.astype(int)
    N_now = 0
    
    printProgressBar(N_now, N_rand_total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    X_train_augmented = X_train
    y_train_augmented = y_train

    for i in range(X_train.shape[0]):
        if n_data_of_classes[y_train[i]] <= 3000: 
            N_rand = np.floor(3000/n_data_of_classes[y_train[i]])
            N_rand = N_rand.astype(int)

        for j in range(N_rand):
            
            N_now += 1
            X_train_PIL = Image.fromarray(X_train[i,:,:,:])
            
            rw = np.floor(random.random()*12 + 18)
            rw = int(rw)
            rs = np.floor((32 - rw) * random.random()) 
            rs = int(rs)

            # randomly crop and reshape to (32,32,3)
            randomly_cropped_image = X_train_PIL.crop((rs,rs,rs+rw,rs+rw))
            distorted_image = randomly_cropped_image.resize((32,32), Image.ANTIALIAS)

            # randomly adjust brightness
            enhancer = ImageEnhance.Brightness(distorted_image)
            distorted_image_ = enhancer.enhance(random.random())

            # convert image to uint8 array
            distorted_image_ = np.array(distorted_image_, dtype=np.uint8)

            # append distorted image on X_train
            distorted_image_ = distorted_image_[np.newaxis,:]
            X_train_augmented = np.append(X_train_augmented, distorted_image_, axis=0)
            y_train_augmented = np.append(y_train_augmented, [y_train[i]], axis=0)
            
            printProgressBar(N_now, N_rand_total, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print("augmented training images are generated.")
    print("%d -> %d" %(X_train.shape[0], X_train_augmented.shape[0]))
if generate_distorted_images == True:
    train_augmented = {'features': X_train_augmented, 'labels': y_train_augmented}

    adata_name = "train_aug.p"
    with open(adata_name, "wb") as f:
        pickle.dump(train_augmented, f)
with open("train_aug.p", mode="rb") as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
print('number of data for each class (augmented dataset):')
show_dataset_summary(train)
def normalize_images(X_):
    return X_ / 255 - 0.5

X_train_ = normalize_images(X_train)
print('training set is normalized')

X_valid_ = normalize_images(X_valid)
print('Validation set is normalized')

X_test_ = normalize_images(X_test)
print('Test set is normalized')
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
x = tf1.placeholder(tf.float64, (None, 32, 32, 3))
x = tf.cast(x, tf.float32)
y = tf1.placeholder(tf.uint8, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf1.placeholder(tf.float32)
rate = 0.001
EPOCHS = 40
BATCH_SIZE = 4096
display_step = 2
save_step = 5

do_train = 1

device_type = "/gpu:0"
def LeNet_he(x):    
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x32.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean=mu, stddev=np.sqrt(2/(5*5*3))))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b

    # batch normalization
    mean_, var_ = tf.nn.moments(conv1, [0,1,2])
    conv1 = tf.nn.batch_normalization(conv1, mean_, var_, 0, 1, 0.0001)


    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x64.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), mean=mu, stddev=np.sqrt(2/(5*5*32))))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b 

    # batch normalization
    mean_, var_ = tf.nn.moments(conv2, [0,1,2])
    conv2 = tf.nn.batch_normalization(conv2, mean_, var_, 0, 1, 0.0001)


    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Flatten. Input = 5x5x64. Output = 1600.
    fc0 = Flatten()(conv2)

    # Layer 3: Fully Connected. Input = 1600. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(1600,120), mean=mu, stddev=np.sqrt(2/(1600))))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b

    # batch normalization
    mean_, var_ = tf.nn.moments(fc1, axes=[0])
    fc1 = tf.nn.batch_normalization(fc1, mean_, var_, 0, 1, 0.0001)

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=np.sqrt(2/120)))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    # batch normalization
    mean_, var_ = tf.nn.moments(fc2, axes=[0])
    fc2 = tf.nn.batch_normalization(fc2, mean_, var_, 0, 1, 0.0001)

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu, stddev=np.sqrt(2/84)))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits

print('LeNet w/ He initialziation is ready')

with tf.device(device_type):
#     logits = LeNet(x, keep_prob)
    logits = LeNet_he(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    
    loss_operation = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess, open('LeNet_He_BatchNorm.csv', 'w') as csvfile:
    fieldnames = ['epoch', 'loss', 'train_acc', 'test_acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # training
    if do_train == 1:
        sess.run(tf.global_variables_initializer())
        
        num_examples = len(X_train_)

        print("Training...")
        print()
        
        # epoch
        for epoch in range(EPOCHS):
            avg_loss = 0.
            total_batch = int(num_examples/BATCH_SIZE)

            X_train_, y_train = shuffle(X_train_, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train_[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                avg_loss += sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})/total_batch

            if epoch % display_step == 0:
                train_acc = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                valid_acc = sess.run(accuracy_operation, feed_dict={x: X_valid_, y: y_valid})
                
                print("Epoch: %03d/%03d, loss: %.9f, train acc: %.3f, valid acc: %.3f" 
                      % (epoch + 1, EPOCHS, avg_loss, train_acc, valid_acc))

            if epoch % save_step == 0:
                saver.save(sess, "nets/traffic_sign_lenet-" + str(epoch))
            writer.writerow({'epoch': epoch + 1, 'loss': avg_loss, 'train_acc': train_acc, 'test_acc': valid_acc})
                
        test_acc = sess.run(accuracy_operation, feed_dict={x: X_test_, y:y_test})        
        print("Test accuracy: %.3f" % (test_acc))
    
    if do_train == 0:
#         epoch = epoch_to_restore
        saver.restore(sess, tf.train.latest_checkpoint('nets/'))
        print("Model restored.")
        
        # calculate training accuracy
        batch_size_for_cal = 10000
        n_train_right = 0
        offset = 0
        tstep = np.floor(X_train_.shape[0]/10000)
        for t in range(tstep.astype(int)):            
            if X_train_.shape[0] - (batch_size_for_cal + offset) < 0:
                batch_size_for_cal = X_train_.shape[0] - offset
            n_train_right += sess.run(accuracy_operation, 
                                     feed_dict={x: X_train_[offset:offset+batch_size_for_cal],
                                                y: y_train[offset:offset+batch_size_for_cal]}) * batch_size_for_cal
        
        train_acc = n_train_right/X_train_.shape[0]
        
        # validation and test accuracy
        valid_acc = sess.run(accuracy_operation, feed_dict={x: X_valid_, y: y_valid})
        test_acc = sess.run(accuracy_operation, feed_dict={x: X_test_, y:y_test})
        print("Train accuracy: %.3f" % (train_acc))
        print("Validation accuracy: %.3f" % (valid_acc))
        print("Test accuracy: %.3f" % (test_acc))
img_set = np.ones(shape=(1,32,32,3))
fig = plt.figure(figsize=(9,3))
n_test_img = 0
img = Image.open('testimg.jpg').convert('RGB')
img = img.resize((32,32), Image.ANTIALIAS) 
        
        # append image as (N, 32, 32, 3) 
img_set[n_test_img,:,:,:] = np.array(img, dtype=np.float32)/1.0
n_test_img += 1
        
ax_ = fig.add_subplot(1, 1, n_test_img)
ax_.imshow(img, cmap="gray")
X_img_test = normalize_images(img_set)

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('nets/'))
    predict_type = sess.run(tf.argmax(logits, 1), feed_dict={x: X_img_test})
    print(predict_type)
print(predict_type.dtype)
print(predict_type[0])
with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('nets/'))
    y_test_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: X_test_})
## configure confussion matrix
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.title(title)
    
    tick_marks = np.arange(len(classes))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    # write the values of confusion matrix
    cm_font ={'size': '7'}
    tick_font = {'size': '8'}
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] >= 0.02:
            plt.text(j, i, '{:.2f}'.format(cm[i, j])[1:],
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     **cm_font)
        
    
    plt.xticks(tick_marks, classes, **tick_font)
    plt.yticks(tick_marks, classes, **tick_font)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.colorbar()
    plt.grid()
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)
classes = [str(i) for i in range(len(SignNames))]

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,7))
plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix', normalize=True)
plt.show()

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps
<<<<<<< HEAD
=======
from base64 import b64decode
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

>>>>>>> master

app = Flask(__name__)
api = Api(app)


testdict = {"test1": "hello1", "test2": "hello2"}
managedict = {'key1': 'val1'}

<<<<<<< HEAD
=======

def Model(x):
    mu = 0
    sigma = 0.1
    keep_prob = 0.9
    strides_conv = [1, 1, 1, 1]
    strides_pool = [1, 2, 2, 1]

    # ________________________________Layer 1__________________________________________________

    # Convolutional. Input = 64x50x1. Filter = 5x5x1. Output = 60x46x6.
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    conv1_W = tf.Variable(tf.random_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=strides_conv, padding='VALID') + conv1_b

    # Apply activation function
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.dropout(conv1, keep_prob)

    # ________________________________Layer 2__________________________________________________

    # Convolutional. Input = 60x28x6. Filter = 3x3x6. Output = 14x14x6.
    conv2_W = tf.Variable(tf.random_normal(shape=(3, 3, 6, 12), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(12))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=strides_conv, padding='SAME') + conv2_b

    # Pooling. Input = 60x28x6. Output = 30x14x6.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # ________________________________Layer 3__________________________________________________

    # Convolutional. Input = 30x14x6. Filter = 5x5x12. Output = 26x10x16.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 16), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(16))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=strides_conv, padding='VALID') + conv3_b

    # Apply activation function
    conv3 = tf.nn.relu(conv3)

    # Pooling. Input = 26x10x16. Output = 13x5x16.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # ________________________________Layer 4__________________________________________________

    # Flatten. Input = 13x5x16. Output = 400.
    fc0 = tf.reshape(conv3, [-1, int(13 * 13 * 16)])
    fc0 = tf.nn.dropout(fc0, keep_prob)

    # Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(13 * 13 * 16, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Apply activation function
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # ________________________________Layer 5__________________________________________________

    # Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Apply activation function
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # ________________________________Layer 6__________________________________________________

    # Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 3), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(3))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 64, 64, 1))
logits = Model(x)
saver = tf.train.Saver()
save_file = './' + 'trained_variables.ckpt' # String addition used for emphasis

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)


>>>>>>> master
class Employees(Resource):
    def get(self):  
        return managedict  # Fetches first column that is Employee ID


class test11(Resource):
    def get(self):
        return "This test worked!!!!2"

    def post(self):
        parser = reqparse.RequestParser()
        args = parser.parse_args()
        print (args)
        return "Ok"

class Employees2(Resource):
    def get(self, keyToAdd, valueToAdd):
        managedict[keyToAdd] = valueToAdd
        return "Added the following to the 'Manage' dictionary:{0}: {1}".format(keyToAdd, valueToAdd)


class testdict1(Resource):
    def get(self, test_id):
        return "{testid}: {dictitem}".format(testid=test_id, dictitem=testdict[test_id])
    def post(self):
        value = "what"
        return value

<<<<<<< HEAD
class value(Resource):
   # @app.route('/post/', methods=['POST'])
    def post(self):
        #json_data = request.get_json(force=True)
        #print(json_data)
        data = request.stream
        print(data)
        return "data"
=======

def get_image(image_id, image_type, bw=1):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname, bw)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    if bw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class value(Resource):
    def post(self):
        data = request.stream
        inp = data.read()
        b64imgfile = open('img.png', 'wb')
        b64imgfile.write(b64decode(inp))
        b64imgfile.close()

        filename = r"img.png"

        im = cv2.imread(filename, 1)
        im_bw = cv2.imread(filename, 0)
        vfunc = np.vectorize(lambda x, y: np.array([x - y]) if x - y >= 0 else np.array([0]))
        dim = (64, 64)
        im = cv2.resize(im, dim)
        red = im[:, :, 2]
        bw = cv2.resize(im_bw, dim)
        res = vfunc(red, bw)
        data = res.reshape((64, 64, 1))
        resp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {x: [data]}
            classification = sess.run(logits, feed_dict)
            print(classification)
            resp = classification.argmax()
        print(resp)
        return dumps({"response": str(resp)})

>>>>>>> master

class dictmanage(Resource):
    def get(self):
        return managedict


api.add_resource(test11, '/test11')
api.add_resource(testdict1, '/testdict1/<string:test_id>')
<<<<<<< HEAD
#api.add_resource(testdict1, '/testdict1')
=======
>>>>>>> master
api.add_resource(value, '/value')


api.add_resource(dictmanage, '/manage')
api.add_resource(Employees, '/employees') # Route_1

api.add_resource(
    Employees2, '/manageadd/<string:keyToAdd>/<string:valueToAdd>')

if __name__ == '__main__':
     app.run(port=5002)

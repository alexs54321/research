import cv2
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from mtcnn import MTCNN
from matplotlib import pyplot
from numpy import asarray, expand_dims
from PIL import Image

"""
https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
"""

def main():
    face_image = "face_db/bradpitt1.jpg"

    # in case you want to show any images, the following commented lines are useful
    # can use cv2.imshow instead of pyplot.imshow, but pyplot.imshow takes RGB triple, and cv2.imshow BGR triple so must use cv2.cvtColor
    #cv2.imshow("Face", cv2.cvtColor(extract_face(face_image), cv2.COLOR_BGR2RGB))

    # pyplot.show() is useless if not using pyplot.imshow
    #pyplot.show()

    #cv2.waitKey(0)

    # alternatively can use "vgg16" or "senet50" as model parameter, by default it is "vgg16"
    model = VGGFace(model="resnet50")

    # see what the model expects as inputs - model expects (224, 224, 3) images
    print("inputs: %s" % model.inputs)

    # see what the model gives as outputs - model will output 1 of 8631 identities (8631 people in MS-Celeb-1M dataset)
    print("outputs: %s" % model.outputs)

    # call defined function below to extract face from image
    face = extract_face(face_image)

    # convert pixel values to float
    face = face.astype("float32")

    # convert single face into many samples
    samples = expand_dims(face, axis=0)

    # get the image ready to be processed by VGGFace model by scaling pixel values in the same way training data was, need version=2 since this is VGGFace2
    samples = preprocess_input(samples, version=2)

    # predict method returns top 5 predictions in unreadable format
    yhat = model.predict(samples)

    # must decode predictions to get [[person_1, probability], [person_2, probability],..., [person_5, probability]]
    results = decode_predictions(yhat)

    # print results to look nice
    for result in results[0]:
        print("%s: %.3f%%" % (result[0], result[1]*100))


# extracts face from given file, outputs as 224x224x3 array (by default)
def extract_face(filename, required_size=(224, 224)):

    # get face image to be used
    face_image = pyplot.imread(filename)

    # mtcnn is slow but performs way better than haar cascades
    detector = MTCNN()

    # use mtcnn to detect the face
    results = detector.detect_faces(face_image)

    # get coordinates from return value of detect_faces, not sure the specifics here but likely not important
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height

    # get face only from face image
    face = face_image[y1:y2, x1:x2]

    # convert array to Pillow image (PIL image) just to resize it
    image = Image.fromarray(face)
    image = image.resize(required_size)

    # convert PIL image back to array after resizings
    face_array = asarray(image)

    return face_array


main()

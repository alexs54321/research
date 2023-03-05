from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from celebrity_face_recognition import extract_face
from scipy.spatial.distance import cosine
from numpy import asarray

"""
https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
"""

def main():
    face_images = ["face_db/bradpitt1.jpg", "face_db/bradpitt2.jpg", "face_db/bradpitt3.jpg", "face_db/samueljackson1.jpg", "face_db/keanureeves1.jpg"]

    # call defined function below to get an array of 2048-dimension vectors, one vector to represent each face
    embeddings = get_embeddings(face_images)
    
    # face embedding can be stored as a definition of a person's identity, compare against other embeddings to verify identity
    brad_pitt_identity = embeddings[0]
    
    # compare Brad Pitt's face embedding vs others and print out result
    for i in range(1, len(embeddings)):
        print("Matching Brad Pitt with %s..." % (face_images[i]))
        is_match(brad_pitt_identity, embeddings[i])


# returns array of face embeddings, one for each face image file, each vector is 2048 dimensions
def get_embeddings(filenames):

    # use extract_face function from face recognition example
    faces = [extract_face(face) for face in filenames]

    # convert faces to numpy array of floats
    samples = asarray(faces, "float32")

    # preprocess faces in same way as before
    samples = preprocess_input(samples, version=2)

    # create model without classifier (include_top=False), define input_shape, and pooling="avg" to get a column vector output instead of 244x244x3 array
    # this model can then be used to get a "face embedding", a vector that represents facial features extracted by the model
    model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")

    # get embeddings for each sample (face), predict method returns numpy array of predictions
    yhat = model.predict(samples)

    return yhat


# calculates cosine distance between two embeddings, returns True if identities match (cosine distance < 0.5 (1.0 is maximum possible distance, minimum possible distance is 0.0))
def is_match(known_embedding, candidate_embedding, thresh=0.5):

    # use SciPy function for calculating cosine distance
    score = cosine(known_embedding, candidate_embedding)

    # print result
    if score <= thresh:
        print("Face is a match (%.3f <= %.3f)" % (score, thresh))
    else:
        print("Face is not a match (%.3f <= %.3f)" % (score, thresh))


main()

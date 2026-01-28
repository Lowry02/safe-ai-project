import torch

from model import *

encoder = EncoderNoPooling()
classifier = LinearClassifier()

encoder_path = ""
classifier_path = ""
encoder.load_state_dict(torch.load(encoder_path, map_location="cuda"))
classifier.load_state_dict(torch.load(classifier_path, map_location="cuda"))

cnn = CNN.import_from(encoder, classifier)
cnn.eval()

torch.save(cnn.state_dict(), "")
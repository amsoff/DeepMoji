# from deepmoji import SentenceTokenizer, finetune_chainthaw, define_deepmoji
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.finetuning import (
    load_benchmark,
    finetune)
from deepmoji.model_def import (
    deepmoji_transfer,
    deepmoji_architecture,
    deepmoji_feature_encoding,
    deepmoji_emojis)


vocab_path = ''
pretrained_path = ''
maxlen = 100
nb_classes = 2
# Load your dataset into two Python arrays, 'texts' and 'labels'

# Splits the dataset into train/val/test sets. Then tokenizes each text into separate words and convert them to our vocabulary.
st = SentenceTokenizer(vocab_path, maxlen)
split_texts, split_labels = st.split_train_val_test(texts, labels)
# Defines the DeepMoji model and loads the pretrained weights
model = deepmoji_transfer(nb_classes, maxlen, pretrained_path)
# Finetunes the model using our chain-thaw approach and evaluates it
model, acc = finetune(model, split_texts, split_labels)
print("Accuracy: {}".format(acc))
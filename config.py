# ----------------------------------- General ----------------------------------- #
SEED = 42

# ----------------------------------- Dataset ----------------------------------- #
RAW_DATA = "./data/reviews.csv"
TRAIN_CSV = "./data/kaggle_dataset/train_balanced_kaggle.csv"
TEST_CSV = "./data/kaggle_dataset/test_balanced_kaggle.csv"

# -------------------------------- Preprocessing -------------------------------- #
REMOVE_PUNCTUATION = True
KEEP_NEGATIONS = True

# -------------------------------- Vectorization -------------------------------- #
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 8000

# ----------------------------------- Training ----------------------------------- #
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# ------------------------------------ Model ------------------------------------ #
INPUT_SIZE = 8000
NUM_CLASSES = 3

# ------------------------------------ Paths ------------------------------------ #
MODEL_NAME = "sentiment_model_medium_larger_dataset"
MODEL_PATH = f"./models/{MODEL_NAME}.pth"
VECTORIZER_PATH = f"./models/{MODEL_NAME}_vectorizer.pkl"
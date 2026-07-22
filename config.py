# ----------------------------------- General ----------------------------------- #
SEED = 42

# ----------------------------------- Dataset ----------------------------------- #
RAW_DATA = "./data/reviews.csv"
TRAIN_CSV = "./data/train_processed.csv"
TEST_CSV = "./data/test_processed.csv"

# -------------------------------- Preprocessing -------------------------------- #
REMOVE_PUNCTUATION = True
KEEP_NEGATIONS = True

# -------------------------------- Vectorization -------------------------------- #
NGRAM_RANGE = (1, 2)
MAX_FEATURES = None

# ----------------------------------- Training ----------------------------------- #
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# ------------------------------------ Model ------------------------------------ #
INPUT_SIZE = 4932
NUM_CLASSES = 2

# ------------------------------------ Paths ------------------------------------ #
MODEL_NAME = "sentiment_model_wider"
MODEL_PATH = f"./models/{MODEL_NAME}.pth"
VECTORIZER_PATH = f"./models/{MODEL_NAME}_vectorizer.pkl"
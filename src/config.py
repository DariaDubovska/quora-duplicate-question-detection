import os


try:
    import google.colab  # type: ignore

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


if IN_COLAB:
    DRIVE_DIR = (
        "/content/drive/MyDrive/"
        "ML Projects/Quora duplicates"
    )
else:
    DRIVE_DIR = (
        r"G:\My Drive\ML Projects\Quora duplicates"
    )


cache_file = os.path.join(
    DRIVE_DIR,
    "embeddings",
    "bert_embeddings.pkl",
)

chunks_dir = os.path.join(
    DRIVE_DIR,
    "embeddings",
    "bert_cache_chunks",
)

feature_file = os.path.join(
    DRIVE_DIR,
    "data",
    "processed",
    "quora_with_bert_cosine_similarity.csv",
)

tfidf_save_path = os.path.join(
    DRIVE_DIR,
    "embeddings",
    "tfidf_text_map_v1.pkl",
)

raw_data_dir = os.path.join(
    DRIVE_DIR,
    "data",
    "raw",
)

train_file = os.path.join(
    raw_data_dir,
    "quora_question_pairs_train.csv.zip",
)

test_file = os.path.join(
    raw_data_dir,
    "quora_question_pairs_test.csv.zip",
)

processed_train_file = os.path.join(
    DRIVE_DIR,
    "data",
    "processed",
    "quora_train_preprocessed.parquet",
)

processed_test_file = os.path.join(
    DRIVE_DIR,
    "data",
    "processed",
    "quora_test_preprocessed.parquet",
)

PROJECT_ROOT_DIR  = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

results_file = os.path.join(
    PROJECT_ROOT_DIR ,
    "reports",
    "model_results.csv",
)
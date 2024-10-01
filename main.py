from src.data.preprocess import preprocess_data


if __name__ == "__main__":
    train_set, test_test, dev_set = preprocess_data()

    X_train, train_desc, y_train = train_set

    
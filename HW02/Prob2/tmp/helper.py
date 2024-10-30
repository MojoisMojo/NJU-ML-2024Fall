if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import r2_score

    DTC = DecisionTreeClassifier(random_state=14)
    data = load_iris()
    X, y = data.data, data.target
    train_features, test_features, train_targets, test_targets = train_test_split(
        X, y, train_size=0.7, shuffle=False, random_state=14
    )
    DTC.fit(train_features, train_targets)
    predict_targets = DTC.predict(test_features)
    right = 0
    all_len = test_targets.shape[0]
    for i in range(all_len):
        if predict_targets[i] == test_targets[i]:
            right += 1
    print("正确率为:", right / all_len)

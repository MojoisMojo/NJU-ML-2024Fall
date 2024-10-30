def get_adult_income_data():
    # 读取训练和测试数据
    data_train = pd.read_csv(
        "../data/adult/adult.data", header=None, skipinitialspace=True
    )
    data_test = pd.read_csv(
        "../data/adult/adult.test", header=None, skiprows=1, skipinitialspace=True
    )
    data = pd.concat([data_train, data_test], ignore_index=True)

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data.columns = columns

    # 分离特征和标签
    X = data.drop("income", axis=1)
    y = data["income"]
    y = y.apply(lambda s: 1 if (s == ">50K" or s == ">50K.") else 0)
    X_prepared = myHotEncoder(X)

    # 将数据分回训练集和测试集
    X_train = X_prepared[: len(data_train)]
    X_test = X_prepared[len(data_train) :]
    y_train = y[: len(data_train)].values
    y_test = y[len(data_train) :].values

    return X_train, X_test, y_train, y_test
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Stacking Classifier",
    layout="wide"
)

st.title("Stacking Classifier")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column],
        default=[
            col for col in df.columns
            if col != target_column
        ][:2]
    )

    st.sidebar.header("Hyperparameter Tuning")

    n_estimators = st.sidebar.slider(
        "Random Forest Trees",
        10,
        200,
        100
    )

    max_depth = st.sidebar.slider(
        "Max Depth",
        1,
        20,
        5
    )

    test_size = st.sidebar.slider(
        "Test Size",
        0.1,
        0.5,
        0.2
    )

    if len(feature_columns) > 0:

        X = df[feature_columns]
        y = df[target_column]

        X = pd.get_dummies(
            X,
            drop_first=True
        )

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42
        )

        rf_grid = GridSearchCV(
            RandomForestClassifier(
                random_state=42
            ),
            {
                "n_estimators":[
                    n_estimators,
                    n_estimators + 50
                ],
                "max_depth":[
                    max_depth,
                    max_depth + 2
                ]
            },
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )

        with st.spinner(
            "Finding Best Parameters..."
        ):
            rf_grid.fit(
                X_train,
                y_train
            )

        best_rf = rf_grid.best_estimator_

        st.subheader("Best Parameters")

        st.write(
            rf_grid.best_params_
        )

        stack_model = StackingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression()
                ),
                (
                    "dt",
                    DecisionTreeClassifier()
                ),
                (
                    "knn",
                    KNeighborsClassifier()
                )
            ],
            final_estimator=best_rf
        )

        stack_model.fit(
            X_train,
            y_train
        )

        y_pred = stack_model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        st.subheader(
            "Model Evaluation"
        )

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        st.subheader(
            "Confusion Matrix"
        )

        fig1, ax1 = plt.subplots(
            figsize=(5,4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1
        )

        st.pyplot(fig1)

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose()
        )

        st.subheader(
            "Feature Distribution"
        )

        selected_feature = st.selectbox(
            "Select Feature",
            feature_columns
        )

        fig2, ax2 = plt.subplots(
            figsize=(6,4)
        )

        sns.histplot(
            df[selected_feature],
            kde=True,
            ax=ax2
        )

        st.pyplot(fig2)

        st.subheader(
            "Manual Prediction"
        )

        input_data = {}

        for col in feature_columns:

            if df[col].dtype == "object":

                input_data[col] = st.selectbox(
                    f"Select {col}",
                    df[col].unique()
                )

            else:

                input_data[col] = st.number_input(
                    f"Enter {col}",
                    value=float(
                        df[col].mean()
                    )
                )

        if st.button(
            "Predict"
        ):

            input_df = pd.DataFrame(
                [input_data]
            )

            input_df = pd.get_dummies(
                input_df
            )

            input_df = input_df.reindex(
                columns=X.columns,
                fill_value=0
            )

            input_scaled = scaler.transform(
                input_df
            )

            prediction = stack_model.predict(
                input_scaled
            )[0]

            predicted_class = label_encoder.inverse_transform(
                [prediction]
            )[0]

            probability = np.max(
                stack_model.predict_proba(
                    input_scaled
                )
            )

            st.success(
                f"Predicted Class: {predicted_class}"
            )

            st.info(
                f"Confidence: {probability:.2f}"
            )

else:

    st.info(
        "Please upload a CSV file to continue"
    )

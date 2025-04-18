{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNwLPkBlbz2WA9CNSvt20AD",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AniruddhaYadav-AI/DM/blob/main/Untitled12.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "tWjrzE7x_I_-"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import graphviz\n",
        "\n",
        "from IPython.display import display\n",
        "from sklearn.datasets import load_breast_cancer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.svm import LinearSVC\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.tree import export_graphviz"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
        "from sklearn.metrics import accuracy_score\n",
        "import numpy as np\n",
        "\n",
        "# Load datasets\n",
        "train_df = pd.read_csv(\"wine_data.csv\")\n",
        "test_df = pd.read_csv(\"wine_data_test.csv\")\n",
        "\n",
        "# Split features and target\n",
        "X = train_df.drop(columns=[\"quality\"])\n",
        "y = train_df[\"quality\"]\n",
        "\n",
        "# 1) Train-validation split (80/20)\n",
        "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)\n",
        "\n",
        "# Initialize and train Random Forest\n",
        "clf = RandomForestClassifier(random_state=0)\n",
        "clf.fit(X_train, y_train)\n",
        "\n",
        "# Training evaluation\n",
        "y_train_pred = clf.predict(X_train)\n",
        "train_acc = accuracy_score(y_train, y_train_pred)\n",
        "train_cm = pd.crosstab(y_train, y_train_pred, rownames=['True'], colnames=['Predicted'], margins=True)\n",
        "print(\"Training Accuracy: {:.4f}\".format(train_acc))\n",
        "print(\"Training Confusion Matrix:\\n\", train_cm)\n",
        "\n",
        "# Validation evaluation\n",
        "y_val_pred = clf.predict(X_val)\n",
        "val_acc = accuracy_score(y_val, y_val_pred)\n",
        "val_cm = pd.crosstab(y_val, y_val_pred, rownames=['True'], colnames=['Predicted'], margins=True)\n",
        "print(\"\\nValidation Accuracy: {:.4f}\".format(val_acc))\n",
        "print(\"Validation Confusion Matrix:\\n\", val_cm)\n",
        "\n",
        "# 2) Dummy test set evaluation\n",
        "X_test = test_df.drop(columns=[\"quality\"])\n",
        "y_test = test_df[\"quality\"]\n",
        "y_test_pred = clf.predict(X_test)\n",
        "test_acc = accuracy_score(y_test, y_test_pred)\n",
        "test_cm = pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True)\n",
        "print(\"\\nTest Accuracy: {:.4f}\".format(test_acc))\n",
        "print(\"Test Confusion Matrix:\\n\", test_cm)\n",
        "\n",
        "# 3) Stratified 10-fold Cross-Validation\n",
        "from sklearn.metrics import confusion_matrix\n",
        "skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)\n",
        "fold_accuracies = []\n",
        "\n",
        "for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):\n",
        "    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]\n",
        "    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]\n",
        "\n",
        "    model = RandomForestClassifier(random_state=0)\n",
        "    model.fit(X_train_fold, y_train_fold)\n",
        "\n",
        "    y_pred_fold = model.predict(X_val_fold)\n",
        "    acc = accuracy_score(y_val_fold, y_pred_fold)\n",
        "    fold_accuracies.append(acc)\n",
        "    print(f\"Fold {fold} Accuracy: {acc:.4f}\")\n",
        "\n",
        "print(f\"\\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f}\")\n",
        "\n",
        "# Optional concise cross-validation using cross_val_score\n",
        "scores = cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=10)\n",
        "print(\"\\nCross-validation scores:\", scores)\n",
        "print(\"Average cross-validation score: {:.4f}\".format(scores.mean()))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ohH7rl2J_ZFB",
        "outputId": "e05b0146-fdc1-42c1-8845-9a0795aef593"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Accuracy: 1.0000\n",
            "Training Confusion Matrix:\n",
            " Predicted    0    1   All\n",
            "True                     \n",
            "0          510    0   510\n",
            "1            0  577   577\n",
            "All        510  577  1087\n",
            "\n",
            "Validation Accuracy: 0.7978\n",
            "Validation Confusion Matrix:\n",
            " Predicted    0    1  All\n",
            "True                    \n",
            "0           95   33  128\n",
            "1           22  122  144\n",
            "All        117  155  272\n",
            "\n",
            "Test Accuracy: 0.9800\n",
            "Test Confusion Matrix:\n",
            " Predicted   0   1  All\n",
            "True                  \n",
            "0          47   1   48\n",
            "1           1  51   52\n",
            "All        48  52  100\n",
            "Fold 1 Accuracy: 0.7279\n",
            "Fold 2 Accuracy: 0.7868\n",
            "Fold 3 Accuracy: 0.8456\n",
            "Fold 4 Accuracy: 0.7574\n",
            "Fold 5 Accuracy: 0.8235\n",
            "Fold 6 Accuracy: 0.8162\n",
            "Fold 7 Accuracy: 0.8824\n",
            "Fold 8 Accuracy: 0.8162\n",
            "Fold 9 Accuracy: 0.7059\n",
            "Fold 10 Accuracy: 0.7852\n",
            "\n",
            "Average Cross-Validation Accuracy: 0.7947\n",
            "\n",
            "Cross-validation scores: [0.83088235 0.77941176 0.81617647 0.82352941 0.79411765 0.77205882\n",
            " 0.77205882 0.86764706 0.78676471 0.82222222]\n",
            "Average cross-validation score: 0.8065\n"
          ]
        }
      ]
    }
  ]
}
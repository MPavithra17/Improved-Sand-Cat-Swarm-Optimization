#HPTFINAL
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Input
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time # Import the time module
# -------------------------------
# Parameters
# -------------------------------
n_splits = 5
n_pop = 30
T_iter = 50
alpha, K, epsilon = 0.2, 0.5, 1e-8
R_min, R_max = 0, 2
BEST_HYPERPARAMS = {}
# -------------------------------
# Helpers
# -------------------------------
def handle_missing_values(df):
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.fillna(num_df.mean())
    cat_df = df.select_dtypes(include=['object']).copy()
    for col in cat_df.columns:
        if not cat_df[col].mode().empty:
            cat_df[col] = cat_df[col].fillna(cat_df[col].mode()[0])
        else:
            cat_df[col] = cat_df[col].fillna("")
    return pd.concat([num_df, cat_df], axis=1)

def encode_features(df, features, target):
    df = df.copy()
    label_encoders = {}
    for col in features + [target]:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    return df, label_encoders


def maybe_scale(X_tr, X_te, model):
    if model in ['KNN','MLP','ANN','CNN','LSTM','SVC']:
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)
    return X_tr, X_te

# -------------------------------
# RF Feature Selection
# -------------------------------
def rf_auto_feature_selection(X, y, plot=False):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    threshold = importance_df['Importance'].mean()
    selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
    print(f"\n🌲 RF auto-selected {len(selected_features)} features")
    return selected_features

# -------------------------------
# Hyperparameters
# -------------------------------
def decode_hyperparameters(model_name, hp_vector):

    if model_name == 'RandomForest':
        return {
            'n_estimators': max(200, int(hp_vector[0]) if len(hp_vector) > 0 else 500),
            'max_depth': max(5, int(hp_vector[1]) if len(hp_vector) > 1 else 10),
            'min_samples_split': max(2, int(hp_vector[2]) if len(hp_vector) > 2 else 2),
            'min_samples_leaf': max(1, int(hp_vector[3]) if len(hp_vector) > 3 else 1),
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        }

    # Keep rest as is
    elif model_name == 'KNN':
        return {'n_neighbors': max(1, int(hp_vector[0] if len(hp_vector)>0 else 5))}

    elif model_name == 'MLP':
        return {'hidden_layer_sizes': (int(hp_vector[0] if len(hp_vector)>0 else 128),
                                       int(hp_vector[1] if len(hp_vector)>1 else 64)),
                'max_iter':300}
    elif model_name == 'CatBoost':
        return {
            'depth': max(2, int(hp_vector[0]) if len(hp_vector) > 0 else 5),
            'learning_rate': max(0.001, float(hp_vector[1]) if len(hp_vector) > 1 else 0.05),
            'iterations': max(50, int(hp_vector[2]) if len(hp_vector) > 2 else 100),
            'verbose': 0
        }


    elif model_name == 'ANN':
        return {
            'layers': max(1, min(int(hp_vector[0]), 3)),
            'neurons': max(16, min(int(hp_vector[1]), 128)),
            'learning_rate': max(0.0005, min(float(hp_vector[2]), 0.01)),
            'batch_size': max(16, min(int(hp_vector[3]), 64)),
            'dropout': max(0.3, min(float(hp_vector[4]), 0.6)),
            'epochs': max(5, min(int(hp_vector[5]), 50))   # 🔥 NEW
        }

    elif model_name == 'LSTM':
        return {
            'units': max(16, min(int(hp_vector[0]), 128)),
            'learning_rate': max(0.0005, min(float(hp_vector[1]), 0.01)),
            'batch_size': max(16, min(int(hp_vector[2]), 64)),
            'dropout': max(0.3, min(float(hp_vector[3]), 0.6)),
            'epochs': max(5, min(int(hp_vector[4]), 50))   # 🔥 NEW
        }


    elif model_name == 'CNN':
        return {
            'filters': max(16, min(int(hp_vector[0]), 128)),
            'kernel': max(2, min(int(hp_vector[1]), 5)),
            'learning_rate': max(0.0005, min(float(hp_vector[2]), 0.01)),
            'batch_size': max(16, min(int(hp_vector[3]), 64)),
            'dropout': max(0.3, min(float(hp_vector[4]), 0.6)),
            'epochs': max(5, min(int(hp_vector[5]), 50))   # 🔥 NEW
        }



    elif model_name == 'SVC':
        return {
            'C': float(hp_vector[0] if len(hp_vector)>0 else 1.0),
            'kernel': 'rbf',
            'probability': True
        }



    elif model_name == 'DT':
        return {
            'max_depth': max(3, int(3 + hp_vector[0] * 8)),      # limit depth
            'min_samples_split': max(10, int(10 + hp_vector[1] * 20)),
            'min_samples_leaf': max(5, int(5 + hp_vector[2] * 10)),
            'criterion': 'entropy',
            'class_weight': 'balanced',
            'random_state': 42
        }


    elif model_name == 'GB':
        return {
            'n_estimators': max(20, int(hp_vector[0]) if len(hp_vector) > 0 else 50),
            'learning_rate': max(0.005, float(hp_vector[1]) if len(hp_vector) > 1 else 0.5),
            'max_depth': max(1, int(hp_vector[2]) if len(hp_vector) > 2 else 2)
        }

    return {}
def get_n_hp(model_name):
    if model_name == 'RandomForest':
        return 6
    elif model_name == 'KNN':
        return 1
    elif model_name == 'CatBoost':
        return 3
    elif model_name in ['ANN']:
        return 6

    elif model_name == 'LSTM':
        return 5
    elif model_name == 'CNN':
        return 6
    elif model_name == 'SVC':
        return 1
    elif model_name == 'DT':
        return 3
    elif model_name == 'GB':
        return 3

    else:
        return 0
# -------------------------------
def fitness_function(individual, X, y, model_name):
    n_features = X.shape[1]
    selected = [f for i,f in enumerate(X.columns) if individual[i]==1]
    if not selected:
        return 0
    X_sel = X[selected].values
    scaler = StandardScaler()
    X_sel_scaled = scaler.fit_transform(X_sel)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    hp_vector = individual[n_features:]
    hp = decode_hyperparameters(model_name, hp_vector)

    for train_idx, test_idx in kf.split(X_sel_scaled, y):
        X_train, X_test = X_sel_scaled[train_idx], X_sel_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        try:
            if model_name == 'RandomForest':
                model = RandomForestClassifier(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif model_name == 'KNN':
                model = KNeighborsClassifier(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif model_name == 'CatBoost':
                hp = decode_hyperparameters('CatBoost', hp_vector)

                if hp['iterations'] <= 0:
                    return 0

                model = CatBoostClassifier(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test).astype(int).flatten()
            elif model_name == 'SVC':
                model = SVC(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif model_name == 'DT':
                model = DecisionTreeClassifier(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif model_name == 'GB':
                model = GradientBoostingClassifier(**hp)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)


            elif model_name in ['ANN']:
                num_classes = len(np.unique(y))
                model = Sequential()
                for _ in range(hp['layers']):
                    model.add(Dense(hp['neurons'], activation='relu'))
                    model.add(Dropout(hp['dropout']))
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(optimizer=Adam(hp['learning_rate']),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
                y_pred = np.argmax(model.predict(X_test), axis=1)

            elif model_name == 'LSTM':
                num_classes = len(np.unique(y))
                X_train_r = X_train.reshape(-1,1,X_train.shape[1])
                X_test_r = X_test.reshape(-1,1,X_test.shape[1])
                model = Sequential([LSTM(hp['units'], input_shape=(1,X_train.shape[1])),
                                    Dropout(0.3),
                                    Dense(num_classes, activation='softmax')])
                model.compile(optimizer=Adam(hp['learning_rate']),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(X_train_r, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
                y_pred = np.argmax(model.predict(X_test_r), axis=1)
            elif model_name == 'CNN':
                num_classes = len(np.unique(y))
                X_train_r = X_train.reshape(-1,X_train.shape[1],1)
                X_test_r = X_test.reshape(-1,X_test.shape[1],1)
                model = Sequential([Conv1D(hp['filters'], hp['kernel'], activation='relu', input_shape=(X_train.shape[1],1)),
                                    MaxPooling1D(2), Dropout(0.3), Flatten(),
                                    Dense(num_classes, activation='softmax')])
                model.compile(optimizer=Adam(hp['learning_rate']),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(X_train_r, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
                y_pred = np.argmax(model.predict(X_test_r), axis=1)
            scores.append(accuracy_score(y_test, y_pred))
        except:
            scores.append(0)
    return np.mean(scores)

def update_position_novel(ind, best, worst, iter, max_iter, alpha_0=0.2, K_0=0.5, epsilon=1e-8):
    alpha = alpha_0 * (1 - iter / max_iter)             # alpha decreases over time
    K = K_0 * np.random.rand()                          # stochastic K

    # Escape term with stochastic factor
    escape = (ind - worst) / (np.abs(worst - ind) + epsilon) * np.random.rand(len(ind))

    # Adaptive cosine spiral frequency
    R = np.random.uniform(0, 2)
    spiral = np.cos(np.pi * R * (1 - iter / max_iter))

    new_ind = ind + K * escape * spiral
    new_ind = np.where(np.random.rand(len(ind)) < alpha, best, new_ind)

    # Binary threshold
    return (new_ind > 0.5).astype(int)


def scso_hpt(X, y, model_name, n_hp=5):
    n_features = X.shape[1]
    population = np.random.rand(n_pop, n_features+n_hp)
    population[:,:n_features] = (population[:,:n_features]>0.5).astype(int)

    for iter_ in range(T_iter):
        fitness = [fitness_function(ind, X, y, model_name) for ind in population]
        best = population[np.argmax(fitness)]
        worst = population[np.argmin(fitness)]

        for i in range(n_pop):
            population[i] = update_position_novel(
                population[i],
                best,
                worst,
                iter=iter_,
                max_iter=T_iter,
                alpha_0=0.2,
                K_0=0.5
            )

    best_ind = population[np.argmax([fitness_function(ind, X, y, model_name) for ind in population])]
    selected_features = [X.columns[i] for i in range(n_features) if best_ind[i]==1]

    # Correlation filtering
    if selected_features:
        corr_matrix = X[selected_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col]>0.95)]
        filtered_features = [f for f in selected_features if f not in to_drop]
        print(f"\n💥 Removed Highly Correlated Features (SCSO): {to_drop}")
        print(f"✅ Filtered Features after SCSO Deduplication: {filtered_features}")
    else:
        filtered_features = []
        # -------------------------------
    # Store BEST hyperparameters globally (DO NOT print here)
    # -------------------------------
    BEST_HYPERPARAMS[model_name] = decode_hyperparameters(
        model_name,
        best_ind[n_features:]
    )
    return filtered_features, best_ind[n_features:]

def train_model_collect_metrics(model_name, X, y, selected_features=None, hp_vector=None):
    if hp_vector is None:
        hp_vector = []
    TOTAL_FEATURES = X.shape[1]

    if selected_features is not None:
        SELECTED_FEATURES = len(selected_features)
        REDUCTION = 100 * (TOTAL_FEATURES - SELECTED_FEATURES) / TOTAL_FEATURES

        print(f"\n📊 Feature Selection Summary:")
        print(f"   Total original features : {TOTAL_FEATURES}")
        print(f"   Selected features       : {SELECTED_FEATURES}")
        print(f"   Feature reduction (%)   : {REDUCTION:.2f}%")
    else:
        print(f"\n📊 Feature Selection Summary:")
        print(f"   Total original features : {TOTAL_FEATURES}")
        print(f"   Selected features       : {TOTAL_FEATURES}")
        print(f"   Feature reduction (%)   : 0.00%")

    X_sel = X[selected_features].values if selected_features else X.values
    y_sel = y.values
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc, prec, rec, f1 = [], [], [], []

    print(f"\n🎯 Model: {model_name}")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_sel, y_sel), 1):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y_sel[train_idx], y_sel[test_idx]

        # Scale if needed
        X_train, X_test = maybe_scale(X_train, X_test, model_name)
        hp = decode_hyperparameters(model_name, hp_vector)

        # ---------- Train model ----------
        if model_name == 'RandomForest':
            model = RandomForestClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == 'KNN':
            model = KNeighborsClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == 'MLP':
            model = MLPClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == 'CatBoost':
            model = CatBoostClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).astype(int).flatten()

        elif model_name == 'SVC':
            model = SVC(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == 'DT':
            model = DecisionTreeClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == 'GB':
            model = GradientBoostingClassifier(**hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)


        elif model_name in ['ANN']:
            num_classes = len(np.unique(y_sel))
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer=Adam(0.001),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)

        elif model_name == 'LSTM':
            num_classes = len(np.unique(y_sel))
            X_train_r = X_train.reshape(-1,1,X_train.shape[1])
            X_test_r = X_test.reshape(-1,1,X_test.shape[1])
            model = Sequential([
                LSTM(64, input_shape=(1,X_train.shape[1])),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer=Adam(0.001),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_train_r, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
            y_pred = np.argmax(model.predict(X_test_r), axis=1)

        elif model_name == 'CNN':
            num_classes = len(np.unique(y_sel))
            X_train_r = X_train.reshape(-1,X_train.shape[1],1)
            X_test_r = X_test.reshape(-1,X_test.shape[1],1)
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1],1)),
                MaxPooling1D(2),
                Dropout(0.3),
                Flatten(),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer=Adam(0.001),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_train_r, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], verbose=0)
            y_pred = np.argmax(model.predict(X_test_r), axis=1)

        # ---------- Compute metrics ----------
        fold_acc = accuracy_score(y_test, y_pred)
        fold_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        fold_rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        fold_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Append
        acc.append(fold_acc)
        prec.append(fold_prec)
        rec.append(fold_rec)
        f1.append(fold_f1)

        # Print fold-wise metrics
        print(f"Fold {fold}: Acc={fold_acc:.4f}, Prec={fold_prec:.4f}, Rec={fold_rec:.4f}, F1={fold_f1:.4f}")

    # Print mean metrics
    print(f"➡️ {model_name} | Mean Acc={np.mean(acc):.4f} | Mean Prec={np.mean(prec):.4f} | Mean Rec={np.mean(rec):.4f} | Mean F1={np.mean(f1):.4f}")


def train_ml_only_hybrids_scso(X, y):
    """
    Train ML-ONLY hybrid models using SCSO-selected features.
    Suitable for SMALL datasets.
    """

    # ML-only hybrid combinations (NO DL)
    hybrids = [
        ('RandomForest', 'KNN'),        # compulsory
        ('RandomForest', 'SVC'),
        ('RandomForest', 'CatBoost'),
        ('DecisionTree', 'RandomForest'),
        ('DecisionTree', 'KNN'),
        ('GradientBoosting', 'RandomForest'),
        ('GradientBoosting', 'KNN')
    ]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fs_model, clf_model in hybrids:
        print(f"\n🚀 Training ML Hybrid: {fs_model} ➜ {clf_model}")
        # -------------------------------------------------
        # 1️⃣ FEATURE SELECTION USING FS MODEL
        # -------------------------------------------------
        fs_hp = get_n_hp(fs_model)
        selected_features, _ = scso_hpt(X, y, fs_model, fs_hp)

        if not selected_features:
            print("⚠️ No features selected. Skipping.")
            continue

        X_fs = X[selected_features]
        y_fs = y.values

        print(f"\n📊 Feature Selection Summary")
        print(f"   Total features   : {X.shape[1]}")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Reduction (%)    : {(1 - len(selected_features)/X.shape[1]) * 100:.2f}%")

        # -------------------------------------------------
        # 2️⃣ CLASSIFIER HYPERPARAMETER TUNING (SCSO)
        # -------------------------------------------------
        clf_hp_dim = get_n_hp(clf_model)
        _, clf_hp_vector = scso_hpt(X_fs, y, clf_model, clf_hp_dim)
        clf_hp = decode_hyperparameters(clf_model, clf_hp_vector)
        BEST_HYPERPARAMS[clf_model] = clf_hp

        print("\n🔧 Optimized Classifier Hyperparameters:")
        for k, v in clf_hp.items():
            print(f"   {k}: {v}")

        acc_folds, prec_folds, rec_folds, f1_folds = [], [], [], []

        # ---------- SCSO Feature Selection using FIRST ML model ----------
        n_hp = get_n_hp(fs_model)
        selected_features, best_hp = scso_hpt(X, y, fs_model, n_hp=n_hp)

        if not selected_features:
            print("⚠️ No features selected. Skipping...")
            continue

        print(f"✅ SCSO Selected Features ({len(selected_features)}): {selected_features}")

        X_sel = X[selected_features].values
        y_sel = y.values

        for fold, (tr, te) in enumerate(kf.split(X_sel, y_sel), 1):
            X_tr, X_te = X_sel[tr], X_sel[te]
            y_tr, y_te = y_sel[tr], y_sel[te]

            # Scale only if needed
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            # ---------- Second ML Classifier ----------
            hp = decode_hyperparameters(clf_model, [])

            if clf_model == 'RandomForest':
                clf = RandomForestClassifier(**hp)
            elif clf_model == 'KNN':
                clf = KNeighborsClassifier(**hp)
            elif clf_model == 'CatBoost':
                clf = CatBoostClassifier(**hp)
            elif clf_model == 'SVC':
                clf = SVC(**hp)
            else:
                continue

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc_folds.append(accuracy_score(y_te, y_pred))
            prec_folds.append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
            rec_folds.append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
            f1_folds.append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

            print(f"Fold {fold}: Acc={acc_folds[-1]:.4f}, "
                  f"Prec={prec_folds[-1]:.4f}, "
                  f"Rec={rec_folds[-1]:.4f}, "
                  f"F1={f1_folds[-1]:.4f}")

        print("-" * 60)
        print(f"{fs_model}+{clf_model} | Mean Acc={np.mean(acc_folds):.4f} | "
              f"Mean Prec={np.mean(prec_folds):.4f} | "
              f"Mean Rec={np.mean(rec_folds):.4f} | "
              f"Mean F1={np.mean(f1_folds):.4f}")
        print("=" * 60)

# ============================================================
# WEIGHTED DYNAMIC CROP SUITABILITY MAPPING
# ============================================================
def dynamic_crop_mapping(
    X, y, df_original, best_features, label_encoders,
    model_name='RandomForest',
    enzyme_cols=None,
    eta=0.50
):
    """
    Weighted dynamic crop suitability mapping.

    S(c|X) = w1 * P(y|X) + w2 * Phi(e,c)

    where:
    P(y|X)   = predicted fertility probability
    Phi(e,c) = enzyme compatibility score
    w1, w2   = dynamic weights
    eta      = suitability threshold
    """

    # -------------------------------
    # 1. Initialize model
    # -------------------------------
    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            **BEST_HYPERPARAMS.get('RandomForest', {})
        )

    elif model_name == 'CatBoost':
        model = CatBoostClassifier(
            **BEST_HYPERPARAMS.get('CatBoost', {})
        )

    else:
        raise ValueError(f"{model_name} not supported for crop mapping")

    # -------------------------------
    # 2. Train model
    # -------------------------------
    model.fit(X[best_features], y)

    # -------------------------------
    # 3. Predict fertility probabilities
    # -------------------------------
    fertility_prob = model.predict_proba(X[best_features])
    max_fertility_prob = np.max(fertility_prob, axis=1)

    fertility_pred = model.predict(X[best_features]).astype(int).flatten()
    decoded_fertility = label_encoders['n_fert_class'].inverse_transform(
        fertility_pred
    )

    # -------------------------------
    # 4. Select enzyme columns
    # -------------------------------
    if enzyme_cols is None:
        enzyme_cols = [
            col for col in X.columns
            if 'enz' in col.lower()
            or 'eea' in col.lower()
            or 'urease' in col.lower()
            or 'phosphatase' in col.lower()
            or 'enzyme' in col.lower()
        ]

    if len(enzyme_cols) == 0:
        raise ValueError(
            "No enzyme columns found. Please manually pass enzyme_cols."
        )

    enzyme_data = X[enzyme_cols].copy()

    # -------------------------------
    # 5. Enzyme compatibility Phi(e,c)
    # Normalization:
    # (e_k - e_min) / (e_max - e_min)
    # -------------------------------
    enzyme_norm = enzyme_data.copy()

    for col in enzyme_cols:
        e_min = enzyme_data[col].min()
        e_max = enzyme_data[col].max()

        if e_max - e_min == 0:
            enzyme_norm[col] = 0
        else:
            enzyme_norm[col] = (enzyme_data[col] - e_min) / (e_max - e_min)

    enzyme_compatibility = enzyme_norm.mean(axis=1)

    # -------------------------------
    # 6. Dynamic weight calculation
    # -------------------------------
    sigma_y = np.var(max_fertility_prob)
    sigma_e = np.var(enzyme_compatibility)

    if sigma_y + sigma_e == 0:
        w1 = 0.5
        w2 = 0.5
    else:
        w1 = sigma_y / (sigma_y + sigma_e)
        w2 = sigma_e / (sigma_y + sigma_e)

    # -------------------------------
    # 7. Suitability score
    # S(c|X) = w1*P(y|X) + w2*Phi(e,c)
    # -------------------------------
    suitability_score = (
        w1 * max_fertility_prob
        + w2 * enzyme_compatibility
    )

    # -------------------------------
    # 8. Decode enzyme label if available
    # -------------------------------
    if 'enz' in X.columns and 'enz' in label_encoders:
        decoded_enzyme = label_encoders['enz'].inverse_transform(
            X['enz'].astype(int)
        )
    else:
        decoded_enzyme = enzyme_compatibility

    # -------------------------------
    # 9. Prepare output table
    # -------------------------------
    crop_map = pd.DataFrame({
        'Crop': df_original['main'].values,
        'Enzyme': decoded_enzyme,
        'Predicted Fertility Level': decoded_fertility,
        'Fertility Probability': max_fertility_prob,
        'Enzyme Compatibility Score': enzyme_compatibility,
        'w1_Fertility_Weight': w1,
        'w2_Enzyme_Weight': w2,
        'Suitability Score': suitability_score
    })

    # -------------------------------
    # 10. Decision rule
    # -------------------------------
    crop_map['Recommendation Status'] = np.where(
        crop_map['Suitability Score'] >= eta,
        'Suitable',
        'Not Suitable'
    )

    crop_map = crop_map.sort_values(
        by='Suitability Score',
        ascending=False
    ).reset_index(drop=True)

    return crop_map


# -------------------------------
# MAIN
# -------------------------------
def main():
    df = pd.read_csv(
        "C:/Users/Pavithra/OneDrive/Desktop/work3data.csv",
        encoding="ISO-8859-1"
    )
    target = 'n_fert_class'
    features = [c for c in df.columns if c != target]

    df = handle_missing_values(df)
    df_encoded, label_encoders = encode_features(df, features, target)
    X, y = df_encoded[features], df_encoded[target]

    print("\n💣 RF Feature Selection")
    rf_features = rf_auto_feature_selection(X, y)
    print(f"✅ RF-selected features: {rf_features}")

    # ------------------ SCSO Feature Selection ------------------
    print("\n🎯 Running SCSO for Feature Selection...")
    model_name = 'RandomForest'  # or any model you want
    n_hp = get_n_hp(model_name)

    # Run SCSO + HPT
    scso_selected_features, best_hp_scso = scso_hpt(X, y, model_name, n_hp=n_hp)
    print(f"✅ SCSO selected features (before correlation filter): {scso_selected_features}")

    # ------------------ Extra Correlation Filtering ------------------
    if scso_selected_features:
        corr_matrix = X[scso_selected_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        filtered_features_scso = [f for f in scso_selected_features if f not in to_drop]
    else:
        to_drop = []
        filtered_features_scso = []

    print(f"💥 Removed Highly Correlated Features: {to_drop}")
    print(f"✅ Filtered Features after correlation check: {filtered_features_scso}")

    models = [ 'RandomForest','KNN','MLP','CatBoost','SVC','DT','GB','CNN','LSTM','ANN']

    for m in models:
        print(f"\n🎯 Training model: {m}")
        n_hp = get_n_hp(m)
        filtered_features, best_hp = scso_hpt(X, y, m, n_hp=n_hp)
        train_model_collect_metrics(m, X, y, filtered_features, best_hp)

    print("\n🔥 Training ML-ONLY HYBRID MODELS (Small Dataset Optimized)...")
    train_ml_only_hybrids_scso(X, y)

    # -------------------------------
    # PRINT ALL SCSO-TUNED HYPERPARAMETERS (ONCE)
    # -------------------------------
    print("\n================ FINAL SCSO-TUNED HYPERPARAMETERS ================")
    for model, hp in BEST_HYPERPARAMS.items():
        print(f"\n🔧 Model: {model}")
        for k, v in hp.items():
            print(f"   {k}: {v}")

        print("\n🌾 DYNAMIC CROP MAPPING OUTPUT (Best Feature Set):")      
        best_features_cb, _ = scso_hpt(
            X, y, 'GB', get_n_hp('GB')
        )
        
        crop_map = dynamic_crop_mapping(
            X=X,
            y=y,
            df_original=df,
            best_features=best_features_cb,
            label_encoders=label_encoders,
            model_name='GB',
            enzyme_cols=['enz'],   # change/add enzyme columns if needed
            eta=0.50
        )
        
        print(crop_map.head(10))
        
        crop_map.to_csv(
            "dynamic_crop_suitability_mapping.csv",
            index=False
        )
        
        print("\n✅ Dynamic crop suitability mapping saved as CSV")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱ TOTAL EXECUTION TIME: {end_time - start_time:.2f} seconds")

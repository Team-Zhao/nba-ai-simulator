from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)

RATING_FEATURE_COUNT = 6

PLAYER_FEATURES = [
    "finishing",
    "shooting",
    "playmaking",
    "defense",
    "rebounding",
    "physical",
    "expectedMinutesRosterAdjusted",
]

TOP_N_PLAYERS = 10

DATA_PATH = Path(
    "data/processed/player_level_game_features.csv"
)

RANDOM_SEED = 42

SEEDS = [
    7,
    21,
    42,
    84,
    123,
]

def set_seed(
    seed=RANDOM_SEED,
):
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

def chronological_split(
    df,
    train_ratio=0.8,
):
    unique_dates = (
        df["gameDate"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(unique_dates)
        * train_ratio
    )

    cutoff_date = (
        unique_dates.iloc[
            split_index
        ]
    )

    train_df = df[
        df["gameDate"]
        < cutoff_date
    ].copy()

    test_df = df[
        df["gameDate"]
        >= cutoff_date
    ].copy()

    return (
        train_df,
        test_df,
        cutoff_date,
    )

def build_explicit_team_features(
    player_features_raw,
):
    ratings = (
        player_features_raw[
            ...,
            :RATING_FEATURE_COUNT
        ]
    )

    minutes = (
        player_features_raw[
            ...,
            -1
        ]
    )

    minute_sums = (
        minutes.sum(
            axis=2,
            keepdims=True,
        )
    )

    minute_sums = np.clip(
        minute_sums,
        1e-6,
        None,
    )

    weights = (
        minutes
        / minute_sums
    )

    team_ratings = (
        ratings
        * weights[
            ...,
            None
        ]
    ).sum(
        axis=2
    )

    home_ratings = (
        team_ratings[
            :,
            0,
            :
        ]
    )

    away_ratings = (
        team_ratings[
            :,
            1,
            :
        ]
    )

    return (
        home_ratings
        - away_ratings
    ).astype(
        np.float32
    )

def get_player_feature_cols(
    df,
):
    feature_cols = []

    for team_prefix in [
        "home",
        "away",
    ]:
        for player_number in range(
            1,
            TOP_N_PLAYERS + 1,
        ):
            for feature in PLAYER_FEATURES:
                col = (
                    f"{team_prefix}"
                    f"_p{player_number}"
                    f"_{feature}"
                )

                if col not in df.columns:
                    raise ValueError(
                        f"Missing feature column: {col}"
                    )

                feature_cols.append(
                    col
                )

    return feature_cols

class PlayerLevelMLP(nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                64,
            ),
            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return (
            self.network(x)
            .squeeze(1)
        )

class PlayerEncoder(nn.Module):
    def __init__(
        self,
        player_feature_dim=7,
        embedding_dim=16,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                player_feature_dim,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                embedding_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        x,
    ):
        return self.network(
            x
        )

class SharedPlayerEncoderModel(
    nn.Module
):
    def __init__(
        self,
        player_feature_dim=7,
        embedding_dim=16,
        players_per_team=10,
    ):
        super().__init__()

        self.player_feature_dim = (
            player_feature_dim
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.players_per_team = (
            players_per_team
        )

        self.player_encoder = (
            PlayerEncoder(
                player_feature_dim=(
                    player_feature_dim
                ),
                embedding_dim=(
                    embedding_dim
                ),
            )
        )

        explicit_team_feature_dim = 6

        # total_embedding_dim = (
        #     2 * embedding_dim
        #     + explicit_team_feature_dim
        # )

        total_embedding_dim = (
            2 * embedding_dim
        )

        self.game_network = (
            nn.Sequential(
                nn.Linear(
                    total_embedding_dim,
                    64,
                ),
                nn.ReLU(),

                nn.Dropout(
                    0.30
                ),

                nn.Linear(
                    64,
                    32,
                ),
                nn.ReLU(),

                nn.Dropout(
                    0.20
                ),

                nn.Linear(
                    32,
                    1,
                ),
            )
        )

    def forward(
        self,
        x,
        raw_minutes,
        team_features,
    ):
        batch_size = x.shape[0]

        # ------------------------------------------
        # Reshape flat game features
        # ------------------------------------------

        players = x.reshape(
            batch_size,
            2,
            self.players_per_team,
            self.player_feature_dim,
        )

        # Shape:
        # [batch, 2, 10, 7]

        # ------------------------------------------
        # Extract expected minutes
        # ------------------------------------------

        minutes = raw_minutes.reshape(
            batch_size,
            2,
            self.players_per_team,
        )

        # Shape:
        # [batch, 2, 10]

        # ------------------------------------------
        # Encode every player
        # ------------------------------------------

        player_embeddings = (
            self.player_encoder(
                players
            )
        )

        # Shape:
        # [batch, 2, 10, embedding_dim]

        # ------------------------------------------
        # Convert minutes into weights
        # ------------------------------------------

        # ------------------------------------------
        # Learned player importance
        # ------------------------------------------

    
        # Shape:
        # [batch, 2, 10]



        # ------------------------------------------
        # Combine learned importance + minutes
        # ------------------------------------------

        # Shape:
        # [batch, 2, 10]
        minute_sums = (
            minutes
            .sum(
                dim=2,
                keepdim=True,
            )
            .clamp(
                min=1e-6
            )
        )

        minute_weights = (
            minutes
            / minute_sums
        )

        team_embeddings = (
            player_embeddings
            * minute_weights.unsqueeze(
                -1
            )
        ).sum(
            dim=2
        )


        # ------------------------------------------
        # Home + Away team embeddings
        # ------------------------------------------

        home_embedding = (
            team_embeddings[
                :,
                0,
                :
            ]
        )

        away_embedding = (
            team_embeddings[
                :,
                1,
                :
            ]
        )

        game_embedding = torch.cat(
            [
                home_embedding,
                away_embedding,
            ],
            dim=1,
        )

        # Shape:
        # [batch, 2 * embedding_dim]

        # ------------------------------------------
        # Predict game
        # ------------------------------------------

        logits = (
            self.game_network(
                game_embedding
            )
            .squeeze(1)
        )

        return logits

def evaluate_model(
    model,
    X,
    minutes,
    team_features,
    y,
):
    model.eval()

    with torch.no_grad():
        logits = model(
            X,
            minutes,
            team_features,
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    y_true = (
        y.cpu()
        .numpy()
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    logloss = log_loss(
        y_true,
        probabilities,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
    }

def split_train_validation(
    train_df,
    validation_ratio=0.15,
):
    unique_dates = (
        train_df["gameDate"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(unique_dates)
        * (1 - validation_ratio)
    )

    validation_cutoff = (
        unique_dates.iloc[
            split_index
        ]
    )

    fit_df = train_df[
        train_df["gameDate"]
        < validation_cutoff
    ].copy()

    validation_df = train_df[
        train_df["gameDate"]
        >= validation_cutoff
    ].copy()

    return (
        fit_df,
        validation_df,
        validation_cutoff,
    )

def build_temporal_folds(
    train_df,
):
    fold_specs = [
        {
            "fit_end": "2025-02-15",
            "validation_start": "2025-02-16",
            "validation_end": "2025-04-13",
        },
        {
            "fit_end": "2025-11-08",
            "validation_start": "2025-11-09",
            "validation_end": "2025-12-05",
        },
        {
            "fit_end": "2025-12-19",
            "validation_start": "2025-12-20",
            "validation_end": "2026-01-29",
        },
    ]

    folds = []

    for spec in fold_specs:
        fit_end = pd.Timestamp(
            spec["fit_end"]
        )

        validation_start = pd.Timestamp(
            spec["validation_start"]
        )

        validation_end = pd.Timestamp(
            spec["validation_end"]
        )

        fit_df = train_df[
            train_df["gameDate"]
            <= fit_end
        ].copy()

        validation_df = train_df[
            (
                train_df["gameDate"]
                >= validation_start
            )
            &
            (
                train_df["gameDate"]
                <= validation_end
            )
        ].copy()

        folds.append(
            (
                fit_df,
                validation_df,
            )
        )

    return folds

def train_one_fold(
    fit_df,
    validation_df,
    feature_cols,
    seed=42,
):
    set_seed(
        seed
    )

    # ------------------------------------------
    # Prepare NumPy arrays
    # ------------------------------------------

    X_fit_raw = (
        fit_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_validation_raw = (
        validation_df[
            feature_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    fit_players_raw = (
        X_fit_raw.reshape(
            -1,
            2,
            TOP_N_PLAYERS,
            len(
                PLAYER_FEATURES
            ),
        )
    )

    validation_players_raw = (
        X_validation_raw.reshape(
            -1,
            2,
            TOP_N_PLAYERS,
            len(
                PLAYER_FEATURES
            ),
        )
    )

    fit_team_features_raw = (
        build_explicit_team_features(
            fit_players_raw
        )
    )

    validation_team_features_raw = (
        build_explicit_team_features(
            validation_players_raw
        )
    )
    team_scaler = StandardScaler()

    fit_team_features_scaled = (
        team_scaler.fit_transform(
            fit_team_features_raw
        )
        .astype(
            np.float32
        )
    )

    validation_team_features_scaled = (
        team_scaler.transform(
            validation_team_features_raw
        )
        .astype(
            np.float32
        )
    )
    fit_team_features = torch.tensor(
        fit_team_features_scaled,
        dtype=torch.float32,
    )

    validation_team_features = torch.tensor(
        validation_team_features_scaled,
        dtype=torch.float32,
    )
    fit_minutes_raw = (
        fit_players_raw[
            ...,
            -1
        ]
        .astype(
            np.float32
        )
    )

    validation_minutes_raw = (
        validation_players_raw[
            ...,
            -1
        ]
        .astype(
            np.float32
        )
    )
    fit_minutes = torch.tensor(
        fit_minutes_raw,
        dtype=torch.float32,
    )

    validation_minutes = torch.tensor(
        validation_minutes_raw,
        dtype=torch.float32,
    )

    y_fit_raw = (
        fit_df[
            "homeWin"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_validation_raw = (
        validation_df[
            "homeWin"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # ------------------------------------------
    # Standardize
    # ------------------------------------------

# ------------------------------------------
# Standardize player features
#
# Fit one common scaler across ALL player
# slots so the shared player encoder sees
# the same feature scale for every player.
# ------------------------------------------

    player_scaler = StandardScaler()

    fit_player_rows = (
        fit_players_raw.reshape(
            -1,
            len(PLAYER_FEATURES),
        )
    )

    validation_player_rows = (
        validation_players_raw.reshape(
            -1,
            len(PLAYER_FEATURES),
        )
)

    fit_player_rows_scaled = (
        player_scaler.fit_transform(
            fit_player_rows
        )
        .astype(
            np.float32
        )
    )

    validation_player_rows_scaled = (
        player_scaler.transform(
            validation_player_rows
        )
        .astype(
            np.float32
        )
    )

    X_fit_scaled = (
        fit_player_rows_scaled.reshape(
            len(fit_df),
            -1,
        )
    )

    X_validation_scaled = (
        validation_player_rows_scaled.reshape(
            len(validation_df),
            -1,
        )
    )

    # ------------------------------------------
    # Convert to tensors
    # ------------------------------------------

    X_fit = torch.tensor(
        X_fit_scaled,
        dtype=torch.float32,
    )

    y_fit = torch.tensor(
        y_fit_raw,
        dtype=torch.float32,
    )

    X_validation = torch.tensor(
        X_validation_scaled,
        dtype=torch.float32,
    )

    y_validation = torch.tensor(
        y_validation_raw,
        dtype=torch.float32,
    )

    # ------------------------------------------
    # DataLoader
    # ------------------------------------------

    fit_dataset = TensorDataset(
        X_fit,
        fit_minutes,
        fit_team_features,
        y_fit,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )

    fit_loader = DataLoader(
        fit_dataset,
        batch_size=64,
        shuffle=True,
        generator=generator,
    )

    # ------------------------------------------
    # Model
    # ------------------------------------------

    model = SharedPlayerEncoderModel(
        player_feature_dim=len(
            PLAYER_FEATURES
        ),
        embedding_dim=16,
        players_per_team=TOP_N_PLAYERS,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    # ------------------------------------------
    # Early stopping
    # ------------------------------------------

    EPOCHS = 100
    PATIENCE = 10

    best_validation_loss = (
        float("inf")
    )

    best_state = None
    best_epoch = None

    epochs_without_improvement = 0

    # ------------------------------------------
    # Training loop
    # ------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0

        for (
            batch_X,
            batch_minutes,
            batch_team_features,
            batch_y,
        ) in fit_loader:

            optimizer.zero_grad()

            logits = model(
                batch_X,
                batch_minutes,
                batch_team_features,
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_X)
            )

        train_loss = (
            total_loss
            / len(fit_dataset)
        )

        validation_results = (
            evaluate_model(
                model,
                X_validation,
                validation_minutes,
                validation_team_features,
                y_validation,
            )
        )

        validation_loss = (
            validation_results[
                "logloss"
            ]
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = {
                name: parameter
                .detach()
                .clone()

                for (
                    name,
                    parameter,
                )
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    # ------------------------------------------
    # Restore best model
    # ------------------------------------------

    model.load_state_dict(
        best_state
    )

    validation_results = (
        evaluate_model(
            model,
            X_validation,
            validation_minutes,
            validation_team_features,
            y_validation,
        )
    )

    return {
        "best_epoch": best_epoch,
        "fit_loss_last_epoch": train_loss,
        "validation_accuracy": (
            validation_results[
                "accuracy"
            ]
        ),
        "validation_logloss": (
            validation_results[
                "logloss"
            ]
        ),
        "validation_brier": (
            validation_results[
                "brier"
            ]
        ),
    }

def main():
    # ------------------------------------------
    # 1. Load dataset
    # ------------------------------------------
    set_seed()
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "gameId": str,
        },
        parse_dates=[
            "gameDate"
        ],
    )

    df["gameId"] = (
        df["gameId"]
        .str.zfill(10)
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    # ------------------------------------------
    # 2. Find the 140 player features
    # ------------------------------------------

    feature_cols = (
        get_player_feature_cols(
            df
        )
    )

    print(
        "Player feature count:",
        len(feature_cols),
    )

    # ------------------------------------------
    # 3. Chronological split
    # ------------------------------------------
    test_model = (
        SharedPlayerEncoderModel(
            player_feature_dim=len(
                PLAYER_FEATURES
            ),
            embedding_dim=16,
            players_per_team=(
                TOP_N_PLAYERS
            ),
        )
    )

    dummy_input = torch.randn(
        5,
        len(feature_cols),
    )
    dummy_minutes = torch.rand(
        5,
        2,
        TOP_N_PLAYERS,
    ) * 36

    dummy_team_features = torch.randn(
        5,
        RATING_FEATURE_COUNT,
    )

    dummy_output = test_model(
        dummy_input,
        dummy_minutes,
        dummy_team_features,
    )

    print()
    print(
        "Shared encoder dummy input:",
        dummy_input.shape,
    )

    print(
        "Shared encoder output:",
        dummy_output.shape,
    )

    (
        train_df,
        test_df,
        cutoff_date,
    ) = chronological_split(
        df
    )
    temporal_folds = (
    build_temporal_folds(
        train_df
    )
)

    print()
    print(
        "Temporal folds"
    )

    for (
        fold_number,
        (
            fold_fit_df,
            fold_validation_df,
        ),
    ) in enumerate(
        temporal_folds,
        start=1,
    ):

        print()
        print(
            f"Fold {fold_number}"
        )

        print(
            "Fit:",
            fold_fit_df[
                "gameDate"
            ].min(),
            "to",
            fold_fit_df[
                "gameDate"
            ].max(),
            "| games:",
            len(
                fold_fit_df
            ),
        )

        print(
            "Validation:",
            fold_validation_df[
                "gameDate"
            ].min(),
            "to",
            fold_validation_df[
                "gameDate"
            ].max(),
            "| games:",
            len(
                fold_validation_df
            ),
        )

        print(
            "Validation home win rate:",
            fold_validation_df[
                "homeWin"
            ].mean(),
        )

    print(
        "Cutoff date:",
        cutoff_date,
    )

    print(
        "Train games:",
        len(train_df),
    )

    print(
        "Test games:",
        len(test_df),
    )

    fold_results = []

    for seed in SEEDS:
        print()
        print(
            "################################"
        )
        print(
            f"Seed {seed}"
        )
        print(
            "################################"
        )

        for (
            fold_number,
            (
                fold_fit_df,
                fold_validation_df,
            ),
        ) in enumerate(
            temporal_folds,
            start=1,
        ):

            result = train_one_fold(
                fit_df=fold_fit_df,
                validation_df=(
                    fold_validation_df
                ),
                feature_cols=feature_cols,
                seed=seed,
            )

            result["seed"] = seed
            result["fold"] = fold_number

            fold_results.append(
                result
            )

            print(
                f"Fold {fold_number}"
                f" | Acc: "
                f"{result['validation_accuracy']:.4f}"
                f" | LogLoss: "
                f"{result['validation_logloss']:.4f}"
                f" | Brier: "
                f"{result['validation_brier']:.4f}"
            )

    results_df = pd.DataFrame(
        fold_results
    )

    print()
    print(
        "=============================="
    )
    print(
        "Multi-Seed Temporal CV Summary"
    )
    print(
        "=============================="
    )

    print(
        "Mean accuracy:",
        results_df[
            "validation_accuracy"
        ].mean(),
    )

    print(
        "Accuracy std:",
        results_df[
            "validation_accuracy"
        ].std(),
    )

    print()

    print(
        "Mean log loss:",
        results_df[
            "validation_logloss"
        ].mean(),
    )

    print(
        "Log loss std:",
        results_df[
            "validation_logloss"
        ].std(),
    )

    print()

    print(
        "Mean Brier:",
        results_df[
            "validation_brier"
        ].mean(),
    )

    print(
        "Brier std:",
        results_df[
            "validation_brier"
        ].std(),
    )
    print()
    print(
        "Mean results by fold:"
    )

    print(
        results_df
        .groupby(
            "fold"
        )[
            [
                "validation_accuracy",
                "validation_logloss",
                "validation_brier",
            ]
        ]
        .mean()
    )
    print()
    print("================================")
    print("Best Epoch Analysis")
    print("================================")

    print(
        results_df[
            ["seed", "fold", "best_epoch"]
        ].to_string(index=False)
    )

    print()
    print(
        "Median best epoch:",
        results_df["best_epoch"].median()
    )

    print(
        "Mean best epoch:",
        results_df["best_epoch"].mean()
    )

    print()
    print(
        "Best epoch by fold:"
    )

    print(
        results_df.groupby("fold")["best_epoch"]
        .agg(["mean", "median", "min", "max"])
    )
#     # ------------------------------------------
#     # 4. Convert pandas -> NumPy
#     # ------------------------------------------

#     (
#         fit_df,
#         validation_df,
#         validation_cutoff,
#     ) = split_train_validation(
#         train_df
#     )
#     print()
#     print(
#         "Fit date range:",
#         fit_df["gameDate"].min(),
#         "to",
#         fit_df["gameDate"].max(),
#     )

#     print(
#         "Validation date range:",
#         validation_df["gameDate"].min(),
#         "to",
#         validation_df["gameDate"].max(),
#     )

#     print(
#         "Test date range:",
#         test_df["gameDate"].min(),
#         "to",
#         test_df["gameDate"].max(),
#     )

#     print()
#     print(
#         "Fit home win rate:",
#         fit_df["homeWin"].mean(),
#     )

#     print(
#         "Validation home win rate:",
#         validation_df["homeWin"].mean(),
#     )

#     print(
#         "Test home win rate:",
#         test_df["homeWin"].mean(),
#     )

#     print(
#         "Fit games:",
#         len(fit_df),
#     )

#     print(
#         "Validation games:",
#         len(validation_df),
#     )

#     print(
#         "Test games:",
#         len(test_df),
#     )

#     X_fit_raw = (
#         fit_df[
#             feature_cols
#         ]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     X_validation_raw = (
#         validation_df[
#             feature_cols
#         ]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     X_test_raw = (
#         test_df[
#             feature_cols
#         ]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     y_fit_raw = (
#         fit_df["homeWin"]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     y_validation_raw = (
#         validation_df["homeWin"]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     y_test_raw = (
#         test_df["homeWin"]
#         .to_numpy(
#             dtype=np.float32
#         )
#     )

#     print()

    

#     # ------------------------------------------
#     # 5. Standardize features
#     # ------------------------------------------

#     scaler = StandardScaler()

#     X_fit_scaled = (
#         scaler.fit_transform(
#             X_fit_raw
#         )
#         .astype(np.float32)
#     )

#     X_validation_scaled = (
#         scaler.transform(
#             X_validation_raw
#         )
#         .astype(np.float32)
#     )

#     X_test_scaled = (
#         scaler.transform(
#             X_test_raw
#         )
#         .astype(np.float32)
#     )

#     # ------------------------------------------
#     # 6. NumPy -> PyTorch Tensor
#     # ------------------------------------------

#     X_fit = torch.tensor(
#         X_fit_scaled,
#         dtype=torch.float32,
#     )

#     y_fit = torch.tensor(
#         y_fit_raw,
#         dtype=torch.float32,
#     )

#     X_validation = torch.tensor(
#         X_validation_scaled,
#         dtype=torch.float32,
#     )

#     y_validation = torch.tensor(
#         y_validation_raw,
#         dtype=torch.float32,
#     )

#     X_test = torch.tensor(
#         X_test_scaled,
#         dtype=torch.float32,
#     )

#     y_test = torch.tensor(
#         y_test_raw,
#         dtype=torch.float32,
#     )


#     # ------------------------------------------
#     # 7. Build DataLoader
#     # ------------------------------------------

#     fit_dataset = TensorDataset(
#         X_fit,
#         y_fit,
#     )

#     generator = (
#         torch.Generator()
#         .manual_seed(
#             RANDOM_SEED
#         )
#     )

#     fit_loader = DataLoader(
#         fit_dataset,
#         batch_size=64,
#         shuffle=True,
#         generator=generator,
#     )
#         # ------------------------------------------
#     # 7. Create neural network
#     # ------------------------------------------

#     model = PlayerLevelMLP(
#         input_dim=len(
#             feature_cols
#         )
#     )

#     print()
#     print(model)

#     criterion = nn.BCEWithLogitsLoss()

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=0.001,
#         weight_decay=1e-4,
#     )
#     # ------------------------------------------
#     # 9. Train model
#     # ------------------------------------------

#     EPOCHS = 100
#     PATIENCE = 10

#     best_validation_loss = float("inf")
#     best_state = None
#     best_epoch = None

#     epochs_without_improvement = 0

#     for epoch in range(
#         1,
#         EPOCHS + 1,
#     ):
#         model.train()

#         total_loss = 0.0

#         for (
#             batch_X,
#             batch_y,
#         ) in fit_loader:

#             optimizer.zero_grad()

#             logits = model(
#                 batch_X
#             )

#             loss = criterion(
#                 logits,
#                 batch_y,
#             )

#             loss.backward()

#             optimizer.step()

#             total_loss += (
#                 loss.item()
#                 * len(batch_X)
#             )

#         train_loss = (
#             total_loss
#             / len(fit_dataset)
#         )

#         validation_results = (
#             evaluate_model(
#                 model,
#                 X_validation,
#                 y_validation,
#             )
#         )

#         validation_loss = (
#             validation_results[
#                 "logloss"
#             ]
#         )

#         if (
#             validation_loss
#             < best_validation_loss
#         ):
#             best_validation_loss = (
#                 validation_loss
#             )

#             best_epoch = epoch

#             best_state = {
#                 name: parameter
#                 .detach()
#                 .clone()

#                 for (
#                     name,
#                     parameter,
#                 )
#                 in model.state_dict().items()
#             }

#             epochs_without_improvement = 0

#         else:
#             epochs_without_improvement += 1

#         if (
#             epoch == 1
#             or epoch % 5 == 0
#         ):
#             print(
#                 f"Epoch {epoch:03d}"
#                 f" | Train loss: "
#                 f"{train_loss:.4f}"
#                 f" | Validation loss: "
#                 f"{validation_loss:.4f}"
#                 f" | Validation accuracy: "
#                 f"{validation_results['accuracy']:.4f}"
#             )

#         if (
#             epochs_without_improvement
#             >= PATIENCE
#         ):
#             print()
#             print(
#                 "Early stopping at epoch:",
#                 epoch,
#             )

#             break
    
#     model.load_state_dict(
#         best_state
#     )
#     print(
#         "Best epoch:",
#         best_epoch,
#     )

#     print(
#         "Best validation log loss:",
#         best_validation_loss,
#     )

#     # ------------------------------------------
#     # 10. Evaluate
#     # ------------------------------------------

#     fit_results = evaluate_model(
#         model,
#         X_fit,
#         y_fit,
#     )

#     validation_results = evaluate_model(
#         model,
#         X_validation,
#         y_validation,
#     )

#     test_results = evaluate_model(
#         model,
#         X_test,
#         y_test,
#     )

#     print()
#     print(
#         "Fit accuracy:",
#         fit_results["accuracy"],
#     )

#     print(
#         "Validation accuracy:",
#         validation_results[
#             "accuracy"
#         ],
#     )

#     print(
#         "Test accuracy:",
#         test_results[
#             "accuracy"
#         ],
#     )

#     print()
#     print(
#         "Test log loss:",
#         test_results[
#             "logloss"
#         ],
#     )

#     print(
#         "Test Brier:",
#         test_results[
#             "brier"
#         ],
#     )


if __name__ == "__main__":
    main()
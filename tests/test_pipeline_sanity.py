import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES_V2,
    CATEGORICAL_FEATURES_V2,
    NUMERIC_FEATURES_V2,
)
from src.features import criar_features_avancadas
from src.utils import avaliar_modelo, encontrar_threshold_otimo


def _sample_raw_df(n_rows=64):
    """Monta um dataframe mínimo compatível com o pipeline de features."""
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("1990-01-01")
    return pd.DataFrame(
        {
            "Data_Nascimento": [base_date + pd.Timedelta(days=int(i * 31)) for i in range(n_rows)],
            "Email_Corporativo": ["corp@empresa.com" if i % 3 == 0 else "" for i in range(n_rows)],
            "Paginas_Visitadas": rng.integers(1, 25, size=n_rows),
            "Tempo_Site_Min": rng.uniform(0.5, 30.0, size=n_rows),
            "Abriu_Email": rng.integers(0, 2, size=n_rows),
            "Clicou_Email": rng.integers(0, 2, size=n_rows),
            "Salario": rng.uniform(1800.0, 35000.0, size=n_rows),
            "Genero": rng.choice(["Masculino", "Feminino"], size=n_rows),
            "Estado_Civil": rng.choice(["Solteiro", "Casado"], size=n_rows),
            "Setor": rng.choice(["Tecnologia", "Serviços", "Saúde"], size=n_rows),
            "Tamanho_Empresa": rng.choice(["1-10", "11-50", "51-200"], size=n_rows),
            "Estado": rng.choice(["SP", "RJ", "MG"], size=n_rows),
            "Origem": rng.choice(["Google Ads", "Indicação", "Meta Ads"], size=n_rows),
            "Dispositivo": rng.choice(["Desktop", "Mobile"], size=n_rows),
            "Sistema_Operacional": rng.choice(["Windows", "Android", "iOS"], size=n_rows),
        }
    )


class TestPipelineSanity(unittest.TestCase):
    def test_feature_engineering_outputs_expected_columns(self):
        df = criar_features_avancadas(_sample_raw_df())
        required = set(CATEGORICAL_FEATURES_V2 + NUMERIC_FEATURES_V2 + BINARY_FEATURES_V2)
        self.assertTrue(required.issubset(set(df.columns)))
        self.assertEqual(df[NUMERIC_FEATURES_V2].isna().sum().sum(), 0)

    def test_preprocessor_transform_shape_and_finite_values(self):
        df = criar_features_avancadas(_sample_raw_df())
        X = df[CATEGORICAL_FEATURES_V2 + NUMERIC_FEATURES_V2 + BINARY_FEATURES_V2]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_FEATURES_V2),
                ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES_V2),
                ("bin", "passthrough", BINARY_FEATURES_V2),
            ]
        )

        X_proc = preprocessor.fit_transform(X)
        self.assertEqual(X_proc.shape[0], len(df))
        self.assertTrue(np.isfinite(X_proc).all())

    def test_metrics_helpers_return_valid_outputs(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.05, 0.85, 0.2, 0.7, 0.6, 0.15, 0.4, 0.9])

        threshold, f1 = encontrar_threshold_otimo(y_true, y_prob)
        self.assertTrue(0 <= threshold <= 1)
        self.assertTrue(0 <= f1 <= 1)

        res = avaliar_modelo("sanity", y_true, y_prob, threshold=0.5)
        self.assertEqual(res["nome"], "sanity")
        self.assertTrue(0 <= res["accuracy"] <= 1)
        self.assertTrue(0 <= res["f1"] <= 1)
        self.assertTrue(0 <= res["auc_roc"] <= 1)
        self.assertTrue(0 <= res["auc_pr"] <= 1)


if __name__ == "__main__":
    unittest.main()

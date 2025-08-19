from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor

def make_model(name="gb"):
    if name == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5))
        ])
    elif name == "gb":
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.7,
            random_state=42,
        )
    else:
        raise ValueError("Unknown model name")
    return model

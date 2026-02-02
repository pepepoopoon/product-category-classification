from product_category_classification.demo_data import make_smoke_data
from product_category_classification.evaluate import evaluate
from product_category_classification.predict import predict
from product_category_classification.train import train


def test_end_to_end(tmp_path) -> None:
    data_path = tmp_path / "products.csv"
    artifact_dir = tmp_path / "artifacts"
    make_smoke_data().to_csv(data_path, index=False)

    metadata = train(data_path, artifact_dir, random_state=42)
    metrics = evaluate(artifact_dir / "model.joblib", artifact_dir / "test.csv")
    prediction = predict(artifact_dir / "model.joblib", "wireless computer mouse", top_k=2)

    assert metadata["split_strategy"].startswith("seller-group")
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["top_2_accuracy"] <= 1.0
    assert len(prediction["candidates"]) == 2
    assert (artifact_dir / "split_manifest.csv").exists()

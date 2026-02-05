from product_category_classification.demo_data import make_smoke_data
from product_category_classification.evaluate import evaluate
from product_category_classification.experiment import run_experiment
from product_category_classification.predict import predict
from product_category_classification.train import train


def test_smoke_generator_supports_learning_curve() -> None:
    small = make_smoke_data(sellers_per_category=3, seed=17)
    large = make_smoke_data(sellers_per_category=8, seed=17)

    assert len(small) == 27
    assert len(large) == 72
    assert make_smoke_data(seed=17).equals(make_smoke_data(seed=17))


def test_experiment_records_seller_disjoint_metrics() -> None:
    baseline = run_experiment(
        sellers_per_category=6,
        data_seed=17,
        split_seed=19,
        model_seed=23,
        hypothesis="Проверить контракт эксперимента",
    )
    baseline_feature_count = baseline["dataset"].pop("feature_count")
    result = run_experiment(
        sellers_per_category=6,
        data_seed=17,
        split_seed=29,
        model_seed=31,
        hypothesis="Сравнить word-only конфигурацию",
        regularization_c=0.5,
        feature_mode="word",
        word_ngram_max=1,
        text_field_mode="title",
        label_noise_rate=0.10,
        text_noise_rate=0.10,
        validation_fraction=0.25,
        test_fraction=0.25,
        baseline=baseline,
    )

    assert result["dataset"]["sellers"] == 18
    assert 0 <= result["test"]["macro_f1"] <= 1
    assert set(result["test"]["per_class"]) >= {"electronics", "home", "sports"}
    assert result["dataset"]["feature_count"] < baseline_feature_count
    assert "test_macro_f1_delta" in result["comparison"]
    assert result["comparison"]["feature_count_delta"] is None
    assert result["dataset"]["noisy_train_labels"] > 0


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

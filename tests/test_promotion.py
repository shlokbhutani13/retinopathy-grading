from retinopathy.promotion import promotion_decision


def metrics(*, aptos_kappa=0.895, aptos_auroc=0.984, idrid_severe=0.20):
    return {
        "aptos": {
            "quadratic_weighted_kappa": aptos_kappa,
            "referable_auroc": aptos_auroc,
        },
        "idrid": {"per_class_recall": [0.7, 0.4, 0.5, idrid_severe, 0.5]},
    }


def test_promotes_when_severe_recall_improves_within_guardrails():
    result = promotion_decision(
        metrics(idrid_severe=0.20),
        metrics(aptos_kappa=0.88, aptos_auroc=0.976, idrid_severe=0.35),
    )

    assert result["promote"] is True
    assert all(result["checks"].values())


def test_rejects_candidate_when_aptos_kappa_loss_exceeds_limit():
    result = promotion_decision(
        metrics(idrid_severe=0.20),
        metrics(aptos_kappa=0.874, idrid_severe=0.35),
    )

    assert result["promote"] is False
    assert result["checks"]["aptos_kappa"] is False


def test_rejects_candidate_when_aptos_auroc_loss_exceeds_limit():
    result = promotion_decision(
        metrics(idrid_severe=0.20),
        metrics(aptos_auroc=0.973, idrid_severe=0.35),
    )

    assert result["promote"] is False
    assert result["checks"]["aptos_referable_auroc"] is False


def test_rejects_candidate_without_external_severe_recall_gain():
    result = promotion_decision(
        metrics(idrid_severe=0.20),
        metrics(idrid_severe=0.20),
    )

    assert result["promote"] is False
    assert result["checks"]["idrid_severe_recall"] is False

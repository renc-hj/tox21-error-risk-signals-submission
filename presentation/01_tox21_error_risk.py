"""One coherent 10-minute Lectrace presentation with four selected states."""

from lectrace import image, text

from _project_data import (
    ROOT,
    build_gcn_forward_demo,
    build_split_presentation_state,
    build_uncertainty_presentation_state,
    load_headline_results,
    load_model_performance,
    load_within_similarity_summary,
)


FIGURES = ROOT / "reports" / "figures"


def main() -> None:
    text("# Can uncertainty identify GCN errors under molecular shift?")
    text(
        "**Healthcare problem.** Toxicity screening must prioritize which compounds "
        "need closer review, "
        "but molecular models can face chemistry unlike their training data. "
        "A confident-looking prediction is therefore not automatically a safe "
        "decision. Here, Tox21 provides 7,607 standardized molecules and 12 "
        "in-vitro assay endpoints—not clinical toxicity labels."
    )
    text(
        "**Motivation.** The supervision is incomplete and imbalanced: each molecule has "
        "labels for 9.95 endpoints on average, and positive prevalence ranges from "
        "2.87% to 16.00%. A model can perform well overall yet provide no direct way "
        "to identify which individual predictions are likely to be wrong. This makes "
        "both prediction quality and reliable error prioritization important."
    )
    text(
        "The **Lecture 11 recording on responsible AI** connected uncertainty "
        "and robustness to distribution shift and asked whether **prediction "
        "errors can be detected before deployment**. We turn that course theme "
        "into two research questions."
    )
    text(
        "**RQ1.** How does test performance of fingerprint-based Logistic "
        "Regression and a multi-task GCN differ across the three fixed splits, "
        "and do these realised splits follow the expected structural-similarity ordering?"
    )
    text(
        "**RQ2.** As test molecules become less similar to training data, do "
        "prediction error and MC Dropout uncertainty increase, and which of "
        "structural distance, single-pass uncertainty, and MC Dropout mutual "
        "information most consistently identifies errors?"
    )
    text(
        "**Approach.** Compare fingerprint Logistic Regression with a three-layer "
        "multi-task GCN under random, scaffold, and similarity-cluster splits. "
        "For the GCN, compare structural distance, single-pass uncertainty, and "
        "50-pass MC Dropout mutual information. The next four states are selected "
        "for insight; they are not the complete training trace."
    )
    text("## STATE 1 — Molecular structure changes the evaluation setting")
    split_state = build_split_presentation_state()  # @inspect split_state @stepover
    text(
        "**What is happening?** The submitted Morgan-fingerprint, Bemis–Murcko "
        "scaffold, and Butina-clustering functions assign a compact molecule set "
        "using the same rules as the full experiment. Scaffold and cluster members "
        "remain together."
    )
    text(
        "**What changes?** The molecule representations stay fixed, but group-aware "
        "assignment reduces close structural overlap between training and test data. "
        "**Why it matters:** the three splits probe different realised forms of "
        "molecular shift instead of relying on one random test set."
    )
    model_performance = load_model_performance()  # @inspect model_performance @stepover
    image(str(FIGURES / "model_performance_by_split.png"), width=850)
    text(
        "**Report Figure 1.** GCN AUROC was **0.789, 0.724, and 0.802** for "
        "random, scaffold, and similarity-cluster. GCN AUROC exceeded LR in all "
        "splits, while LR AUPRC was higher for random and scaffold and nearly "
        "equal for similarity-cluster. Performance showed no consistent split ordering. "
        "Because each strategy used one fixed partition, these are realised-partition "
        "comparisons rather than a general split effect."
    )

    text("## STATE 2 — A molecular graph becomes 12 endpoint predictions")
    gcn_state = build_gcn_forward_demo()  # @inspect gcn_state @stepover
    text(
        "**What is happening?** Atom-level feature vectors pass through three graph "
        "convolution layers. Each layer updates an atom representation using "
        "information from neighboring atoms; global mean pooling then creates one "
        "molecule representation, and the multi-task head produces 12 endpoint probabilities."
    )
    text(
        "**What changes?** For the three-atom example, the inspected shapes move from "
        "$3\\times166$ node features through three $3\\times128$ hidden states, to a "
        "$1\\times128$ pooled molecule representation and $1\\times12$ endpoint "
        "probabilities. **Why it matters:** the trained models use this same architecture "
        "before their outputs become risk signals. This compact execution uses fixed "
        "demonstration weights; it is not a reported experimental prediction."
    )
    text("## STATE 3 — Predictions become two uncertainty signals")
    uncertainty_state = build_uncertainty_presentation_state()  # @inspect uncertainty_state @stepover
    text(
        "**What is happening?** Single-pass uncertainty is $0.5-|p-0.5|$. MC "
        "Dropout mutual information instead uses 50 stochastic forward passes. "
        "Observed errors use the same fixed 0.5 decision threshold as the report."
    )
    text(
        "**What changes?** Endpoint probabilities become an error indicator, BCE, "
        "and two competing uncertainty scores. **Why it matters:** the question is "
        "not only whether a molecule is structurally novel, but whether a score can "
        "place actual prediction mistakes near the top of a review queue."
    )
    text(
        "**Illustrative endpoint.** The compact example has true label 1 and "
        "deterministic probability $p=0.48$, so the fixed 0.5 threshold predicts 0 "
        "and produces an error. Its single-pass uncertainty is 0.48 because the "
        "prediction is close to the boundary, while MC Dropout MI instead measures "
        "disagreement across stochastic passes. This is an explanatory example, "
        "not a selected test case from the reported experiment."
    )
    image(str(FIGURES / "similarity_error_uncertainty.png"), width=850)
    text(
        "**Report Figure 2.** Structural distance had almost no mean relationship "
        "with endpoint BCE or MC Dropout MI. Similarity bins showed local variation "
        "but no stable monotonic pattern across splits."
    )
    text("## STATE 4 — Aggregate the error-ranking evidence")
    headline_results = load_headline_results()  # @inspect headline_results @stepover
    within_bin_results = load_within_similarity_summary()  # @inspect within_bin_results @stepover
    text(
        "**What is happening?** Error-detection metrics aggregate the full "
        "endpoint–seed results, then paired comparisons repeat the test within broad "
        "similarity bins."
    )
    text(
        "**What changes?** We move from an inspectable example to the tracked full "
        "experiment and then control coarsely for structural similarity. **Why it "
        "matters:** this tests whether uncertainty contributes information beyond "
        "a simple novelty score."
    )
    image(str(FIGURES / "error_risk_signal_comparison.png"), width=850)
    text(
        "**Report Figure 3.** Error-detection AUROC for single-pass versus MI was "
        "**0.778 vs 0.669** (random), **0.708 vs 0.630** (scaffold), and "
        "**0.787 vs 0.697** (similarity-cluster). Bars are means across 36 "
        "endpoint–seed cells; error bars are SD, not confidence intervals."
    )
    text(
        "**Report Table 5.** Within broad similarity bins, cells favouring "
        "single-pass were **95.1%**, **89.5%**, and **90.6%**. Mean AUROC "
        "differences were 0.118, 0.066, and 0.090; every 95% bootstrap confidence "
        "interval excluded zero."
    )
    text("## Key result and healthcare interpretation")
    text(
        "Single-pass uncertainty ranked errors better than MC Dropout MI in these "
        "fixed experiments, including within broad similarity bins. The result is "
        "**threshold-aligned** because both the score and error indicator depend on "
        "the same 0.5 threshold; it does not establish superior epistemic uncertainty."
    )
    text(
        "In early-stage screening, the score could prioritize predictions for further "
        "experimental testing or expert review; it would not replace either one or "
        "certify a compound as safe. Its practical utility was not evaluated. Tox21 "
        "measures in-vitro assay activity, so this study **does not establish clinical "
        "toxicity or drug safety** and requires repeated partitions plus external or "
        "temporally separated validation."
    )
if __name__ == "__main__":
    main()

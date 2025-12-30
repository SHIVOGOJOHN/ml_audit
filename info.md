# Reproducible Interactive Experiments: Audit Trails for ML Preprocessing
## Simple Explanation
Many interactive analyses (cleaning, filtering, imputing missing values) are performed through point-and-click tools and never recorded. If a colleague asks "how did you clean that data?", you can't reproduce it exactly. Your research would automatically capture every interactive step (filter rows, impute mean, one-hot encode) into an audit trail that can be replayed to regenerate the exact dataset. Then measure: how much do unreproducible steps affect final model results?.
​

## Mathematical Foundation
Mathematical Foundation
Data Provenance as DAG: Represent pipeline as directed acyclic graph:
G=(V,E) where V={operations},E=dependenciesG=(V,E) where V={operations},E=dependencies
Example: v1=v1= "read CSV", v2=v2= "filter age > 18", v3=v3= "impute mean on NaN"
Reproducibility Verification: Rerun entire DAG with fixed random seeds:
Dout′=Replay(G,Din,seed)=?DoutDout′=Replay(G,Din,seed)=?Dout
Model Variance Due to Irreproducibility:
Varirrep=Erandom seeds[Var(model accuracy)]Varirrep=Erandom seeds[Var(model accuracy)]

## Practical Example
Real-world Application: Data scientist shares a cleaned dataset with a statistician; need to verify exactly which rows were kept/removed, which values imputed.

# Research Contributions
Gap Addressed: Interactive prep steps not recorded; reproducibility impossible
Novel Contribution: Automatic op-log DAG + code export + impact quantification on model variance
Expected Impact: Full data lineage tracing; variance reduction from irreproducible steps
Publication Venues: VLDB, SIGMOD, Nature Machine Intelligence, Data Science and Engineering

Keywords for Literature Review
Data provenance, reproducibility, audit trails, ML pipelines, data lineage

1. What problem this project is actually about

The core problem is not missing logs.
The core problem is epistemic uncertainty in data preparation.

In real workflows, data preprocessing happens like this:

Someone filters rows interactively

Someone clicks “impute mean”

Someone removes columns “that looked noisy”

Someone reruns a model and reports accuracy

Later:

A collaborator asks how the data was prepared

The author cannot reproduce the dataset exactly

The model result cannot be independently verified

This is not a tooling inconvenience.
This is a scientific validity problem.

If the data cannot be regenerated exactly, then:

The model result is not falsifiable

Variance caused by preprocessing is invisible

Reported accuracy mixes signal with undocumented human decisions

Your project targets that exact failure mode.

2. The simple idea in human terms

You want a system that:

Watches every interactive preprocessing action

Records it automatically, without relying on memory or discipline

Replays those actions to regenerate the exact same dataset

Measures how different preprocessing paths change model outcomes

That is it. Everything else is implementation detail.

3. What “reproducible interactive experiments” really means

“Interactive” means:

Steps happen one at a time

Often manually

Often conditionally

Often outside scripts

“Reproducible” means:

The exact same data can be regenerated

Using the same inputs

With the same decisions

In the same order

Most ML pipelines solve scripted reproducibility.
Almost none solve interactive reproducibility.

That is the gap.

4. Data provenance as a DAG, explained simply

You model preprocessing as a graph of transformations, not a notebook.

Nodes

Each node is a transformation:

Read CSV

Filter rows where age >= 21

Impute mean on income

One hot encode category

These are operations, not files.

Edges

Edges encode dependency and order:

You cannot impute before loading

You cannot normalize before imputing

You cannot encode categories after dropping the column

This structure forces logical correctness.

Why a DAG matters

A list says what happened.
A DAG explains why it had to happen in that order.

That difference is what reviewers care about.

5. Replay is not re-running code

This is a critical conceptual point.

Replay means:

Starting from the original raw dataset

Applying the same transformations

With the same parameters

In the same order

With controlled randomness

Replay does not mean “try to recreate something similar”.
Replay means deterministic regeneration.

If replay fails, reproducibility fails.

6. Reproducibility verification, explained clearly

After replay, you compare:

The replayed dataset

The original final dataset

Not by shape.
Not by column names.
By content.

You compute a hash over the entire dataframe including row order.

If hashes match, reproducibility holds.
If hashes differ, something is undocumented or unstable.

This turns reproducibility into a binary, testable property.

7. Where randomness enters and why it matters

Some preprocessing steps involve randomness:

Train-test splits

Stochastic imputers

Sampling

Model initialization

Your system fixes random seeds during replay.

That lets you isolate variance caused by:

Randomness

Versus undocumented preprocessing choices

Without this separation, variance analysis is meaningless.

8. Measuring impact of irreproducibility

This is the part that elevates the work beyond logging.

You ask:
“If preprocessing were slightly different, how much would the model change?”

You:

Create multiple preprocessing variants

Each variant is fully reproducible

Train the same model on each variant

Measure accuracy variance

Mathematically:

You estimate variance of model performance induced by preprocessing differences

Not by model randomness

This directly connects data provenance to statistical reliability.

9. Why this is not sklearn pipelines

A skeptic will say:
“Pipelines already exist.”

They miss the point.

Pipelines:

Assume scripted transformations

Do not capture human intent

Do not explain why steps happened

Do not measure epistemic variance

Your system:

Captures intent declaratively

Records interactive human decisions

Makes preprocessing inspectable

Quantifies downstream uncertainty

Different problem. Different contribution.

10. What the research contribution really is
Gap addressed

Interactive preprocessing decisions are invisible and unreproducible.

Novel contribution

An automatic, declarative, replayable audit trail that:

Captures preprocessing as a DAG

Enables deterministic replay

Verifies reproducibility at the data level

Quantifies model variance induced by preprocessing

Why this matters scientifically

You move preprocessing from:
“Art practiced by individuals”
to
“Object of measurement and verification”

That shift is publishable.

11. Real-world intuition

Imagine a statistician receiving a cleaned dataset.

They ask:

Which rows were removed?

Why those rows?

What values were imputed?

With which statistics?

In what order?

Your system answers all of that automatically, and provably.

Without it, trust is social.
With it, trust is computational.
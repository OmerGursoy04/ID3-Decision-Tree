# ID3 Decision Tree - Binary features & labels
Implemented from scratch a binary decision tree classifier (ID3) using information gain (mutual information) with entropy as splitting criterion.
Features and labels are binary (0/1), input data is a TSV file with a header. The tree supports a max depth limit and tie-breaks equal information gain by choosing the smallest column index.

Features:
- ID3 training with entropy and information gain
- Depth constraint (max_depth)
- Majority vote leaves (ties → predict 1)
- Deterministic tie-breaking on features (smallest index)
- Pretty-print of the learned tree in autograder-compatible format
- CLI for training, predicting, and writing metrics

Data format:
- File type: .tsv
- Header: feature names + label column last
- Rows: binary (0/1)

Implementation Highlights:
- Load data: TSV → Python 2D list of features + labels & a separate list for attribute names
- Mutual Information: MI(X_i; Y) = H(Y) − H(Y | X_i).
- Splitting Criterion: binary split on X_i ∈ {0,1}; stop if pure, empty, depth limit, no attributes, or non-positive information gain.
- Tie-break: on equal gain, pick the smallest column index.
- Prediction: Recursive traversal (DFS) to a leaf; leaf holds vote (majority label).

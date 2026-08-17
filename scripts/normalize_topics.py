import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path("dataset/cleaned_dataset.jsonl")
OUTPUT_FILE = Path("dataset/normalized_dataset.jsonl")


# ============================================================
# TOPIC KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    "NLP": [

        "nlp",
        "natural language",
        "natural language processing",

        "tokenization",
        "token",

        "language model",
        "language modeling",
        "language models",

        "transformer",
        "transformers",
        "transformer models",
        "transformer architecture",

        "attention",
        "attention mechanisms",
        "self-attention",
        "multi-head attention",

        "positional encoding",

        "text generation",
        "text preprocessing",

        "sentiment",
        "sentiment analysis",

        "neural machine translation",

        "word embedding",
        "word embeddings",
        "pretrained embeddings",

        "context vectors",

        "beam search",

        "encoder-decoder",
        "encoder-decoder architecture",
        "encoder-decoder network",

        "sequence-to-sequence",
        "sequence to sequence",
        "seq2seq",

        "language modeling"
    ],


    # --------------------------------------------------------
    # PYTORCH / TENSORFLOW
    # --------------------------------------------------------

    "PyTorch/TensorFlow": [

        # PyTorch
        "pytorch",
        "torch",
        "torch.tensor",
        "torch.nn",
        "torch.nn.module",
        "torch.nn.parameter",
        "torch.utils",
        "torch.utils.data",
        "torch.utils.data.dataloader",
        "dataloader",
        "dataset",

        "torch.save",
        "torch.load",
        "state dict",
        "state_dict",

        "to(device)",
        "cuda",
        "gpu acceleration",
        "gpu acceleration",
        "gpu management",

        "tensor operations",
        "tensors",
        "tensor data types",

        "autodiff",
        "automatic differentiation",
        "autograd",
        "torch.autograd.grad",

        "type conversion",

        "subclassing api",
        "sub-classing module",

        "computational graph",

        # TensorFlow / Keras
        "tensorflow",
        "tensorflow serving",
        "tensorflow model serving",

        "keras",
        "keras implementation",

        "tf.",
        "tf.data",
        "tf.variable",
        "tf.gradienttape",

        "autograph",
        "@tf.function",
        "static graph conversion",
        "graphs vs eager execution",

        "vertex ai",
        "vertex ai prediction service",

        "distribution strategies api",

        "tensorboard",

        "model saving/loading",
        "saving/loading custom models",

        "custom training loops",

        "custom layers",
        "custom models",
        "custom initializers"
    ],


    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    "Deep Learning": [

        "deep learning",
        "deep networks",
        "deep network",

        "neural network",
        "neural networks",

        "neural network architecture",
        "neural network architectures",
        "network architecture",

        # CNN
        "cnn",
        "convolution",
        "convolutional",
        "convolutional layers",

        "filters",
        "feature maps",

        "pooling",
        "pooling layers",
        "average pooling",
        "max pooling",

        "stride",
        "padding",
        "downsampling",

        # CNN architectures
        "resnet",
        "resnet-34",
        "vgg",
        "vggnet",
        "alexnet",
        "googlenet",
        "xception",
        "senet",
        "lenet-5",

        # RNN
        "rnn",
        "simple rnn",
        "recurrent",
        "recurrent layers",
        "recurrent neurons",

        "lstm",
        "long short-term memory",
        "memory cells",

        "gru",
        "gated recurrent units",

        "bptt",
        "backpropagation through time",

        "stateful rnn",
        "deep rnn",

        "long sequences",
        "temporal dependencies",

        # GAN
        "gan",
        "gans",
        "generative adversarial",
        "generative adversarial networks",
        "deep convolutional gan",
        "stylegan",
        "progressive growing gans",
        "gan architectures",

        # Autoencoders
        "autoencoder",
        "autoencoders",
        "variational autoencoder",
        "variational autoencoders",
        "vae",
        "convolutional autoencoders",
        "denoising autoencoders",
        "sparse autoencoders",
        "stacked autoencoders",

        # Object Detection
        "yolo",
        "object detection",
        "object tracking",
        "real-time detection",
        "real time detection",
        "bounding boxes",
        "anchor boxes",
        "region proposal networks",
        "non-max suppression",
        "single-shot detection",
        "tracking-by-detection",

        # Segmentation
        "image segmentation",
        "semantic segmentation",
        "pixel-wise classification",
        "u-net",

        # Neural network training
        "backpropagation",
        "manual backpropagation",

        "activation",
        "activation functions",
        "custom activation functions",

        "relu",
        "leaky relu",
        "sigmoid",
        "tanh",
        "swish",
        "elu",

        "dropout",

        "batch normalization",

        "weight initialization",
        "weight initialization techniques",
        "model initialization",
        "custom initializers",
        "glorot initialization",
        "xavier",
        "he initialization",

        "gradient clipping",
        "vanishing gradient",
        "vanishing gradients",
        "exploding gradient",
        "exploding gradients",
        "unstable gradients",

        "mixed precision training",
        "amp",
        "automatic mixed precision",

        "skip connections",
        "residual blocks",

        "model pre-training",
        "pretrained models",
        "pretrained layers",

        "fine-tuning",
        "fine tuning"
    ],


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    "Machine Learning": [

        "machine learning",
        "machine learning basics",
        "machine learning fundamentals",
        "machine learning models",
        "machine learning algorithms",
        "machine learning problem types",
        "machine learning challenges",

        # Learning types
        "supervised learning",
        "unsupervised learning",
        "semi-supervised learning",

        "reinforcement learning",
        "active learning",

        "transfer learning",

        # Classification
        "classification",
        "binary classification",
        "multiclass classification",
        "multilabel classification",
        "multioutput classification",

        # Regression
        "regression",
        "linear regression",
        "logistic regression",
        "polynomial regression",
        "ridge regression",
        "lasso regression",
        "elastic net regression",
        "regularized linear models",
        "regression models",
        "regression trees",
        "svm regression",

        # Algorithms
        "decision tree",
        "decision trees",

        "random forest",
        "random forests",

        "extra-trees",

        "svm",
        "support vector machine",
        "support vector machines",
        "support vectors",

        "knn",
        "k-nearest neighbors",
        "nearest neighbors",

        # Clustering
        "k-means",
        "kmeans",
        "clustering",
        "cluster",
        "number of clusters",

        "dbscan",
        "mean shift",
        "spectral clustering",
        "hierarchical clustering",
        "agglomerative clustering",

        "gaussian mixtures",
        "bayesian gaussian mixture models",

        # Ensemble
        "ensemble",
        "ensemble methods",
        "ensemble learning",
        "bagging",
        "boosting",
        "adaboost",
        "gradient boosting",
        "random forest",
        "stacking",
        "voting classifiers",
        "random patches",
        "random subspaces",
        "pasting",

        # Optimization
        "gradient descent",
        "batch gradient descent",
        "stochastic gradient descent",
        "sgd",
        "mini-batch gradient descent",

        "optimizers",
        "optimizer",
        "learning rate",

        "adam",
        "adamw",
        "rmsprop",
        "adagrad",
        "adamax",
        "nadam",
        "momentum",
        "nesterov accelerated gradient",

        # Model evaluation
        "cross-validation",
        "cross validation",
        "k-fold cross-validation",

        "model evaluation",
        "model selection",
        "model generalization",

        "performance measures",
        "performance metrics",
        "evaluation metrics",

        "precision",
        "recall",
        "f1-score",
        "roc curve",
        "confusion matrix",

        # Overfitting
        "overfitting",
        "underfitting",
        "generalization",
        "bias-variance tradeoff",

        # Regularization
        "regularization",
        "regularization techniques",
        "l1 regularization",
        "l2 regularization",
        "ridge",
        "lasso",
        "max-norm regularization",

        # Features
        "feature engineering",
        "feature selection",
        "feature scaling",
        "feature transformation",
        "feature extraction",

        # Dimensionality reduction
        "dimensionality reduction",
        "pca",
        "principal components",
        "principal component",
        "randomized pca",
        "incremental pca",
        "random projection",

        "t-sne",
        "stochastic neighbor embedding",
        "manifold learning",
        "isomap",
        "lle",
        "multidimensional scaling",
        "mds",

        # Anomaly detection
        "anomaly detection",
        "novelty detection",
        "isolation forest",
        "one-class svm",
        "local outlier factor",
        "lof",
        "elliptic envelope",
        "fast-mcd",

        # RL
        "q-learning",
        "deep q-learning",
        "dqn",
        "double dqn",
        "dueling dqn",
        "approximate q-learning",
        "policy gradients",
        "policy search",
        "reward optimization",
        "exploration policies",
        "temporal difference learning",
        "prioritized experience replay",

        # Hyperparameters
        "hyperparameter",
        "hyperparameters",
        "hyperparameter tuning",
        "grid search",
        "randomized search",
        "randomized search",
        "model optimization"
    ],


    # --------------------------------------------------------
    # DATA SCIENCE
    # --------------------------------------------------------

    "Data Science": [

        "data science",
        "data analysis",
        "data exploration",
        "data cleaning",
        "data preparation",
        "data preprocessing",
        "preprocessing",
        "preprocessing techniques",

        "data visualization",
        "visualization",

        "data transformation",
        "feature transformation",

        "data compression",

        "statistical",
        "statistical learning theory",

        "correlation",
        "variance",
        "explained variance ratio",

        "forecasting",
        "time series forecasting",
        "time series",
        "multivariate time series",

        "arma",
        "multi-step forecasting",

        "data download",

        "data mismatch",
        "insufficient data",
        "poor quality data",

        "nonrepresentative data",
        "irrelevant features"
    ],


    # --------------------------------------------------------
    # AI FUNDAMENTALS
    # --------------------------------------------------------

    "AI Fundamentals": [

        "artificial intelligence",
        "ai fundamentals",
        "ai types",

        "machine intelligence",

        "intelligent agent",
        "intelligent agents",

        "knowledge representation",

        "expert system",
        "expert systems",

        "reasoning",

        "search algorithm",
        "search algorithms",

        "learning types",
        "learning type",
        "learning paradigms",

        "generative models",
        "generative model",

        "discriminative model",
        "discriminative models",

        "explainable ai",
        "model interpretability"
    ],


    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    "Python": [

        "python",
        "python programming",

        "numpy",
        "pandas",
        "matplotlib",

        "scikit-learn",
        "sklearn"
    ],


    # --------------------------------------------------------
    # CODING / DEBUGGING
    # --------------------------------------------------------

    "Coding/Debugging": [

        "coding",
        "debugging",
        "debugging models",

        "error",
        "errors",

        "exception",
        "exceptions",

        "code",
        "programming"
    ]
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# FIND MAJOR CATEGORY
# ============================================================

def get_category(topic, question, answer):

    topic_text = normalize_text(topic)

    combined_text = normalize_text(
        topic + " " + question + " " + answer
    )

    # --------------------------------------------------------
    # First check the topic itself
    # --------------------------------------------------------

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            keyword = normalize_text(keyword)

            if keyword in topic_text:
                return category

    # --------------------------------------------------------
    # If topic doesn't match, check question + answer
    # --------------------------------------------------------

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            keyword = normalize_text(keyword)

            if keyword in combined_text:
                return category

    # --------------------------------------------------------
    # No match
    # --------------------------------------------------------

    return "Other"


# ============================================================
# MAIN DATASET NORMALIZATION
# ============================================================

def normalize_dataset():

    category_counts = Counter()

    total = 0

    # Create dataset directory if required
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read cleaned dataset
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Read JSON
            # ------------------------------------------------

            try:

                item = json.loads(line)

            except json.JSONDecodeError:

                continue

            # ------------------------------------------------
            # Get fields
            # ------------------------------------------------

            topic = item.get("topic", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            level = item.get("level", "")

            # ------------------------------------------------
            # Find category
            # ------------------------------------------------

            category = get_category(
                topic,
                question,
                answer
            )

            # ------------------------------------------------
            # Create normalized record
            # ------------------------------------------------

            new_item = {

                "question": question,

                "answer": answer,

                "topic": topic,

                "category": category,

                "level": level
            }

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            outfile.write(
                json.dumps(
                    new_item,
                    ensure_ascii=False
                ) + "\n"
            )

            # ------------------------------------------------
            # Count
            # ------------------------------------------------

            category_counts[category] += 1

            total += 1

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("TOPIC NORMALIZATION COMPLETE")
    print("=" * 60)

    print(f"\nTotal examples: {total}")

    print("\nMAJOR CATEGORY DISTRIBUTION")
    print("-" * 60)

    for category, count in category_counts.most_common():

        percentage = (count / total) * 100

        print(
            f"{category:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    print("\n" + "-" * 60)
    print("OUTPUT FILE")
    print("-" * 60)

    print(OUTPUT_FILE)

    print("\nDone!")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    normalize_dataset()
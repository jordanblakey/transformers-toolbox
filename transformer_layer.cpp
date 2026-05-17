#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using Matrix = std::vector<std::vector<float>>;

// Helper function to print matrices beautifully
void printMatrix(const std::string& name, const Matrix& M) {
    std::cout << "\n--- " << name << " ---\n";
    for (const auto& row : M) {
        for (float val : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << val << " ";
        }
        std::cout << "\n";
    }
}

// 1. Raw C++ Matrix Multiplication Core (The absolute foundation of AI)
Matrix matmul(const Matrix& A, const Matrix& B, bool transposeB = false) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    int targetRows = rowsA;
    int targetCols = transposeB ? rowsB : colsB;
    int innerDim = colsA;

    Matrix C(targetRows, std::vector<float>(targetCols, 0.0f));

    for (int i = 0; i < targetRows; ++i) {
        for (int j = 0; j < targetCols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < innerDim; ++k) {
                float valB = transposeB ? B[j][k] : B[k][j];
                sum += A[i][k] * valB;
            }
            C[i][j] = sum;
        }
    }
    return C;
}

// 2. Softmax function applied to matrix rows to normalize attention weights
void softmaxRows(Matrix& M) {
    for (auto& row : M) {
        float maxVal = -INFINITY;
        for (float val : row) if (val > maxVal) maxVal = val; // Stability trick

        float sum = 0.0f;
        for (float& val : row) {
            val = std::exp(val - maxVal);
            sum += val;
        }
        for (float& val : row) {
            val /= sum;
        }
    }
}

int main() {
    // Let's assume a sequence of 3 tokens (e.g., "The cat sat"), each with an embedding size of 4.
    int seq_len = 3;
    int d_k = 4;

    // Simulated hidden states (Token embeddings)
    Matrix X = {
        {1.0f, 0.0f, 2.0f, 0.5f},  // Token 1
        {0.0f, 2.0f, 1.0f, 0.0f},  // Token 2
        {1.5f, 0.5f, 0.0f, 1.0f}   // Token 3
    };

    // Simulated weight projection matrices (In PyTorch, these are learned parameters)
    Matrix W_q = {{0.5f, 0.1f, 0.2f, 0.0f}, {0.0f, 0.6f, 0.1f, 0.3f}, {0.2f, 0.0f, 0.5f, 0.1f}, {0.1f, 0.2f, 0.0f, 0.7f}};
    Matrix W_k = {{0.4f, 0.2f, 0.0f, 0.1f}, {0.1f, 0.5f, 0.2f, 0.0f}, {0.0f, 0.1f, 0.6f, 0.2f}, {0.3f, 0.0f, 0.1f, 0.5f}};
    Matrix W_v = {{0.7f, 0.0f, 0.1f, 0.3f}, {0.1f, 0.8f, 0.0f, 0.1f}, {0.2f, 0.1f, 0.9f, 0.0f}, {0.0f, 0.3f, 0.2f, 0.6f}};

    std::cout << "🚀 Projecting Input Embeddings into Q, K, V spaces..." << std::endl;
    Matrix Q = matmul(X, W_q);
    Matrix K = matmul(X, W_k);
    Matrix V = matmul(X, W_v);

    // 3. Calculate Raw Attention Scores (Q multiplied by the transpose of K)
    Matrix scores = matmul(Q, K, true);

    // 4. Scale the scores by 1 / sqrt(d_k) to prevent exploding gradients
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            scores[i][j] *= scale;
        }
    }

    // 5. Apply Softmax to turn scores into clean attention probabilities
    softmaxRows(scores);
    printMatrix("Attention Matrix (Softmax Probabilities)", scores);

    // 6. Multiply Attention weights by the Value matrix to get final Context vector
    Matrix output = matmul(scores, V);
    printMatrix("Final Layer Output Tensor", output);

    return 0;
}
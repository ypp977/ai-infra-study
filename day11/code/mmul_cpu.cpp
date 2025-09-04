#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

void mmul_cpu(const vector<float>& A, const vector<float>& B, vector<float>& C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    int N = 1024;
    vector<float> A(N * N, 1.0f), B(N * N, 1.0f), C(N * N, 0.0f);

    auto start = high_resolution_clock::now();
    mmul_cpu(A, B, C, N);

    auto end = high_resolution_clock::now();

    cout << "CPU done, C[0]=" << C[0]
         << ", time =" << duration_cast<milliseconds>(end - start).count() << " ms\n";

    return 0;
}

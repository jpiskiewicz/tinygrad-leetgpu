from tinygrad.tensor import Tensor

# A, B, C are tensors on the GPU
def solve(A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int):
    C.replace(A.matmul(B))


if __name__ == "__main__":
  A = Tensor([[1.0, 2.0, 3.0]])
  B = Tensor([[4.0], [5.0], [6.0]])
  C = Tensor.zeros((1, 1))
  solve(A, B, C, A.shape[0], A.shape[1], B.shape[1])
  print(C.numpy())
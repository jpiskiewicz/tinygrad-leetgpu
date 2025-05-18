from tinygrad.tensor import Tensor

# A, B, C are tensors on the GPU
def solve(A: Tensor, B: Tensor, C: Tensor, N: int): # N is useless since tinygrad is smort
  C.replace(A.add(B))

if __name__ == "__main__":
  A = Tensor([1.5, 1.5, 1.5])
  B = Tensor([2.3, 2.3, 2.3])
  C = Tensor.zeros(3)
  solve(A, B, C, 3)
  print(C.numpy())

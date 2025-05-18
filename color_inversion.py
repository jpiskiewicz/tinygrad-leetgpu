from tinygrad.tensor import Tensor

# image is a tensor on the GPU
def solve(image: Tensor, width: int, height: int):
  return Tensor([*([255] * 3), 0]).expand(height, width, -1).sub(image.reshape(height, width, 4)).reshape(width*height*4).mul(Tensor([*([1] * 3), -1] * width * height))

if __name__ == "__main__":
  a = Tensor([255, 0, 128, 255, 0, 255, 0, 255])
  width = 1
  height = 2
  print("Example 1:", solve(a, width, height).numpy())

  a = Tensor([10, 20, 30, 255, 100, 150, 200, 255])
  width = 2
  height = 1
  print("Example 2:", solve(a, width, height).numpy())

  a = Tensor([255, 0, 128, 255, 0, 255, 0, 255])
  width = 1
  height = 2
  print("Example 3:", solve(a, width, height).numpy())
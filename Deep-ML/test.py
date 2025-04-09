
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
  # Return a list where each element is the dot product of a row of 'a' with 'b'.
  # If the number of columns in 'a' does not match the length of 'b', return -1.
  coulumn_threshold = len(a[0])
  if coulumn_threshold != len(b):
      return -1
  result = []
  for row in a:
      if len(row) != coulumn_threshold:
          return -1
      result.append(sum(map(lambda x: x[0]*x[1], zip(row, b))))
  return result


print(matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]))
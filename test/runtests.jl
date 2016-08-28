using GPUArrays
using GPUArrays.CLBackend
using Base.Test

const cl = CLBackend
cl.init()

function test(N)
  a = rand(Float32, N);
  b = rand(Float32, N);
  out = zeros(Float32, N);
  cl_out = cl.CLArray(zeros(Float32, N));
  cl_a = cl.CLArray(a);
  cl_b = cl.CLArray(b);
  @time broadcast!(+, out, a, b);
  @time broadcast!(+, cl_out, cl_a, cl_b);
  @test out == Array(cl_out)
  println("...................")
  @time broadcast!(*, out, a, b);
  @time broadcast!(*, cl_out, cl_a, cl_b);
  @test out == Array(cl_out)

  @test broadcast(min, a, b) == Array(broadcast(min, cl_a, cl_b))
  println("#####################")
end

test(10^6)
test(10^6)
test(10^6)
test(10^6)

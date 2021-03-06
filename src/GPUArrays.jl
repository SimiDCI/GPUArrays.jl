module GPUArrays

abstract Context

include("arrays.jl")
export buffer, context

include(joinpath("backends", "backends.jl"))
export is_backend_supported, supported_backends


end # module

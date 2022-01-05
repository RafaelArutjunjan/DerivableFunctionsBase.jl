module DerivableFunctionsBase


using ForwardDiff, FiniteDifferences
using OffsetArrays # To avoid error thrown in _array_for() when using Symbolics.jacobian()
using Symbolics


# Add graceful errors by implementing _GetGrad(::Val) methods
# Implement FiniteDiff in addition to FiniteDifferences


include("Utils.jl")
export GetArgLength


include("DFunctions.jl")
export DFunction, DerivableFunction
export EvalF, EvaldF, EvalddF, In, Out, InOut


include("DifferentiationOperators.jl")
export diff_backends
export GetDeriv, GetGrad, GetJac, GetHess, GetMatrixJac, GetDoubleJac
export GetGrad!, GetJac!, GetHess!, GetMatrixJac!


end
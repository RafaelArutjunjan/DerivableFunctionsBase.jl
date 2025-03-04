module DerivableFunctionsReverseDiffExt

using DerivableFunctionsBase, ReverseDiff

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


## out-of-place operator backends
_GetDeriv(ADmode::Val{:ReverseDiff}; kwargs...) = throw("GetDeriv() not available for ReverseDiff.jl")
_GetGrad(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient
_GetJac(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian
_GetHess(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian

## in-place operator backends
_GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
_GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
_GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ReverseDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array

import DerivableFunctionsBase: suff
suff(x::ReverseDiff.TrackedReal) = typeof(x)


__init__() = (push!(DerivableFunctionsBase.AvailableBackEnds, :ReverseDiff);  sort!(DerivableFunctionsBase.AvailableBackEnds))

end # module
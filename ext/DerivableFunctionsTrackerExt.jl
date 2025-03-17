module DerivableFunctionsTrackerExt

using DerivableFunctionsBase, Tracker

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


## out-of-place operator backends
_GetDeriv(ADmode::Val{:Tracker}; kwargs...) = throw("GetDeriv() not available for Tracker.jl")
_GetGrad(ADmode::Val{:Tracker}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Tracker.gradient(Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:Tracker}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Tracker.jacobian(Func, p; kwargs...)
_GetHess(ADmode::Val{:Tracker}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Tracker.hessian(Func, p; kwargs...)
_GetDoubleJac(ADmode::Val{:Tracker}; kwargs...) = throw("GetDoubleJac() not available for Tracker.jl")


## Fake in-place operator backends
function _GetGrad!(ADmode::Val{:Tracker}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetGrad!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceGrad!(Y::AbstractVector,F::Function,X::AbstractVector) = copyto!(Y, _GetGrad(ADmode; kwargs...)(F, X))
end
function _GetJac!(ADmode::Val{:Tracker}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceJac!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetJac(ADmode; kwargs...)(F, X))
end
function _GetHess!(ADmode::Val{:Tracker}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetHess!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceHess!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetHess(ADmode; kwargs...)(F, X))
end
function _GetMatrixJac!(ADmode::Val{:Tracker}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetMatrixJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceMatrixJac!(Y::AbstractArray,F::Function,X::AbstractVector) = (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X)))
end

import DerivableFunctionsBase: suff
suff(x::T) where T<:Tracker.TrackedReal = T


import DerivableFunctionsBase: _add_backend
__init__() = _add_backend(:Tracker)

end # module
module DerivableFunctionsZygoteExt

using DerivableFunctionsBase, Zygote

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


## out-of-place operator backends
_GetDeriv(ADmode::Val{:Zygote}; kwargs...) = throw("GetDeriv() not available for Zygote.jl")
_GetGrad(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.gradient(Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.jacobian(Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.hessian(Func, p; kwargs...)
_GetDoubleJac(ADmode::Val{:Zygote}; kwargs...) = throw("GetDoubleJac() not available for Zygote.jl") # Zygote does not support mutating arrays


# Fake in-place methods
function _GetGrad!(ADmode::Val{:Zygote}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetGrad!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceGrad!(Y::AbstractVector,F::Function,X::AbstractVector) = copyto!(Y, _GetGrad(ADmode; kwargs...)(F, X))
end
function _GetJac!(ADmode::Val{:Zygote}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceJac!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetJac(ADmode; kwargs...)(F, X))
end
function _GetHess!(ADmode::Val{:Zygote}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetHess!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceHess!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetHess(ADmode; kwargs...)(F, X))
end
function _GetMatrixJac!(ADmode::Val{:Zygote}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetMatrixJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceMatrixJac!(Y::AbstractArray,F::Function,X::AbstractVector) = (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X)))
end

__init__() = (push!(DerivableFunctionsBase.AvailableBackEnds, :Zygote);  sort!(DerivableFunctionsBase.AvailableBackEnds))

end # module
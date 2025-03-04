module FiniteDifferencesExt

using DerivableFunctionsBase, FiniteDifferences

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


## out-of-place operator backends
_GetDeriv(ADmode::Val{:FiniteDifferences}; kwargs...) = throw("GetDeriv() not available for FiniteDifferences.jl")
_GetGrad(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.grad(central_fdm(order,1), Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:FiniteDifferences}; order::Int=5, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), z->FiniteDifferences.grad(central_fdm(order,1), Func, z)[1], p)[1]


# Fake in-place methods
function _GetGrad!(ADmode::Val{:FiniteDifferences}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetGrad!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceGrad!(Y::AbstractVector,F::Function,X::AbstractVector) = copyto!(Y, _GetGrad(ADmode; kwargs...)(F, X))
end
function _GetJac!(ADmode::Val{:FiniteDifferences}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceJac!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetJac(ADmode; kwargs...)(F, X))
end
function _GetHess!(ADmode::Val{:FiniteDifferences}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetHess!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceHess!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetHess(ADmode; kwargs...)(F, X))
end
function _GetMatrixJac!(ADmode::Val{:FiniteDifferences}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetMatrixJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceMatrixJac!(Y::AbstractArray,F::Function,X::AbstractVector) = (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X)))
end

__init__() = (push!(DerivableFunctionsBase.AvailableBackEnds, :FiniteDifferences);  sort!(DerivableFunctionsBase.AvailableBackEnds))

end # module
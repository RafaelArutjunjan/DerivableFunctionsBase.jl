module DerivableFunctionsFiniteDiffExt

using DerivableFunctionsBase, FiniteDiff

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


## out-of-place operator backends
_GetDeriv(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative
_GetGrad(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient
_GetJac(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_jacobian
_GetHess(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian
_GetDoubleJac(ADmode::Val{:FiniteDiff}; kwargs...) = throw("GetDoubleJac() not available for FiniteDiff.jl")


## in-place methods
#_GetDeriv!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative!
_GetGrad!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient!
function _GetJac!(ADmode::Val{:FiniteDiff}; kwargs...)
    function FiniteDiff__finite_difference_jacobian!(Y::AbstractArray{<:Number}, F::Function, X, args...; kwargs...)
        # in-place FiniteDiff operators assume that function itself is also in-place
        if DerivableFunctionsBase.MaximalNumberOfArguments(F) > 1
            FiniteDiff.finite_difference_jacobian!(Y, F, X, args...; kwargs...)
        else
            # Use fake method
            (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X, args...)))
            # FiniteDiff.finite_difference_jacobian!(Y, (Res,x)->copyto!(Res,F(x)), args...; kwargs...)
        end
    end
end
_GetHess!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian!
_GetMatrixJac!(ADmode::Val{:FiniteDiff}; kwargs...) = _GetJac!(ADmode; kwargs...)


__init__() = (push!(DerivableFunctionsBase.AvailableBackEnds, :FiniteDiff);  sort!(DerivableFunctionsBase.AvailableBackEnds))

end # module
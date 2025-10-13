


global AvailableBackEnds = [:ForwardDiff, :Symbolic]

"""
Shows the currently loaded differentation backends available for use with `DerivableFunctions.jl`.
"""
diff_backends() = copy(DerivableFunctionsBase.AvailableBackEnds)
_add_backend(S::Symbol) = (push!(DerivableFunctionsBase.AvailableBackEnds, S);  sort!(DerivableFunctionsBase.AvailableBackEnds))



GetDeriv(ADmode::Symbol, args...; kwargs...) = GetDeriv(Val(ADmode), args...; kwargs...)
GetGrad(ADmode::Symbol, args...; kwargs...) = GetGrad(Val(ADmode), args...; kwargs...)
GetJac(ADmode::Symbol, args...; kwargs...) = GetJac(Val(ADmode), args...; kwargs...)
GetHess(ADmode::Symbol, args...; kwargs...) = GetHess(Val(ADmode), args...; kwargs...)
GetDoubleJac(ADmode::Symbol, args...; kwargs...) = GetDoubleJac(Val(ADmode), args...; kwargs...)
GetMatrixJac(ADmode::Symbol, args...; kwargs...) = GetMatrixJac(Val(ADmode), args...; kwargs...)


# ForwardDiff as standard
GetDeriv(F::Function, args...; kwargs...) = GetDeriv(Val(:ForwardDiff), F, args...; kwargs...)
GetGrad(F::Function, args...; kwargs...) = GetGrad(Val(:ForwardDiff), F, args...; kwargs...)
GetJac(F::Function, args...; kwargs...) = GetJac(Val(:ForwardDiff), F, args...; kwargs...)
GetHess(F::Function, args...; kwargs...) = GetHess(Val(:ForwardDiff), F, args...; kwargs...)
GetDoubleJac(F::Function, args...; kwargs...) = GetDoubleJac(Val(:ForwardDiff), F, args...; kwargs...)
GetMatrixJac(F::Function, args...; kwargs...) = GetMatrixJac(Val(:ForwardDiff), F, args...; kwargs...)


"""
    GetDeriv(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the scalar derivative of `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
Derivative = GetDeriv(Val(:ForwardDiff), x->exp(-x^2))
Derivative(5.0)
```
For available backends, see `diff_backends()`.
"""
function GetDeriv(ADmode::Val, F::Function, args...; kwargs...)
    Dv = _GetDeriv(ADmode; kwargs...)
    EvaluateDeriv(X::Number, args...) = Dv(F, X, args...)
    EvaluateDeriv(X::SymbolicScalar) = _GetDerivPass(F, X)
end
function GetDeriv(ADmode::Val{:Symbolic}, F::Function, args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, 1, :derivative; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetDeriv(Val(:ForwardDiff), F)
    else M end
end
GetDeriv(ADmode::Val; Kwargs...) = (Dv=_GetDeriv(ADmode; Kwargs...);    EvaluateDerivative(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetDeriv(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)


"""
    GetGrad(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the gradient of `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
Gradient = GetGrad(Val(:ForwardDiff), x->x[1]^2 - x[2]^3)
Gradient(rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetGrad(ADmode::Val, F::Function, args...; kwargs...)
    Dv = _GetGrad(ADmode; kwargs...)
    EvaluateGradient(X::AbstractVector{<:Number}, args...) = Dv(F, X, args...)
    EvaluateGradient(X::AbstractVector{<:SymbolicScalar}) = _GetGradPass(F, X)
end
function GetGrad(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :gradient; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetGrad(Val(:ForwardDiff), F, m, args...)
    else M end
end
GetGrad(ADmode::Val; Kwargs...) = (Dv=_GetGrad(ADmode; Kwargs...);  EvaluateGradient(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetGrad(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)


"""
    GetJac(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the Jacobian of `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
Jacobian = GetJac(Val(:ForwardDiff), x->[x[1]^2, -x[2]^3, x[1]*x[2]])
Jacobian(rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetJac(ADmode::Val, F::Function, args...; kwargs...)
    Dv = _GetJac(ADmode; kwargs...)
    EvaluateJacobian(X::AbstractVector{<:Number}, args...) = Dv(F, X, args...)
    EvaluateJacobian(X::AbstractVector{<:SymbolicScalar}) = _GetJacPass(F, X)
end
function GetJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :jacobian; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetJac(Val(:ForwardDiff), F, m, args...)
    else M end
end
GetJac(ADmode::Val; Kwargs...) = (Dv=_GetJac(ADmode; Kwargs...);  EvaluateJacobian(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)


"""
    GetHess(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the Hessian of `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
Hessian = GetHess(Val(:ForwardDiff), x->x[1]^2 -x[2]^3 + x[1]*x[2])
Hessian(rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetHess(ADmode::Val, F::Function, args...; kwargs...)
    Dv = _GetHess(ADmode; kwargs...)
    EvaluateHess(X::AbstractVector{<:Number}, args...) = Dv(F, X, args...)
    EvaluateHess(X::AbstractVector{<:SymbolicScalar}) = _GetHessPass(F, X)
end
function GetHess(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :hessian; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetHess(Val(:ForwardDiff), F, m, args...)
    else M end
end
GetHess(ADmode::Val; Kwargs...) = (Dv=_GetHess(ADmode; Kwargs...);  EvaluateHessian(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetHess(ADmode::Val, F::DFunction, args...; Kwargs...) = EvalddF(F)



_MakeTuple(Tup::Int) = (Tup,);    _MakeTuple(Tup::Tuple) = Tup
function _SizeTuple(F::Function, m::Int)
    T = try size(F(rand(m))) catch; size(F(rand())) end
    _MakeTuple(T)
end
"""
    GetMatrixJac(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the Jacobian of an array-valued function `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
Jacobian = GetMatrixJac(Val(:ForwardDiff), x->[x[1]^2 x[2]^3; x[1]*x[2] 2])
Jacobian(rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetMatrixJac(ADmode::Val, F::Function, m::Int=GetArgLength(F), f::Tuple=_SizeTuple(F,m), args...; kwargs...)
    Dv = _GetJac(ADmode; kwargs...)
    EvaluateMatrixJacobian(X::AbstractVector{<:Number}, args...) = reshape(Dv(vec∘F, X, args...), f..., m)
    EvaluateMatrixJacobian(X::Number) = reshape(Dv(vec∘F∘(z::AbstractVector->z[1]), [X]), f..., m)
    EvaluateMatrixJacobian(X::Union{<:SymbolicScalar,<:AbstractVector{<:SymbolicScalar}}) = _GetMatrixJacPass(F, X)
end
function GetMatrixJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), f::Tuple=_SizeTuple(F,m), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :matrixjacobian; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetMatrixJac(Val(:ForwardDiff), F, m, f, args...)
    else M end
end
GetMatrixJac(ADmode::Val; Kwargs...) = (Dv=_GetMatrixJac(ADmode; Kwargs...);    EvaluateMatrixJacobian(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetMatrixJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)

# For emergencies: needs an extra evaluation of function to determine length(Func(p))
function _GetMatrixJac(ADmode::Val; kwargs...)
    Dv = _GetJac(ADmode; kwargs...)
    Functor(Func::Function, X::AbstractVector{<:Number}, args...) = reshape(Dv(vec∘Func, X, args...), size(Func(X))..., length(X))
    Functor(Func::Function, X::Number, args...) = reshape(Dv(vec∘Func∘(z::AbstractVector->z[1]), [X], args...), size(Func(X))..., 1)
end


"""
    GetDoubleJac(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes the Jacobian of the Jacobian for a vector-valued function `F` out-of-place via a backend specified by `ADmode`.

Example:
```julia
DoubleJacobian = GetDoubleJac(Val(:ForwardDiff), x->[x[1]^2, x[1]*x[2]^3])
DoubleJacobian(rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetDoubleJac(ADmode::Val, F::Function, m::Int=GetArgLength(F), f::Int=length(F(rand(m))), args...; kwargs...)
    Dv = _GetJac(ADmode; kwargs...)
    if f == 1
        EvaluateDoubleJac(X::AbstractVector{<:Number}) = reshape(Dv(vec∘(z->Dv(F,z)), X), m, m)
        EvaluateDoubleJac(X::AbstractVector{<:SymbolicScalar}) = _GetDoubleJacPass(F, X)
    else
        EvaluateDoubleJacobian(X::AbstractVector{<:Number}) = reshape(Dv(vec∘(z->Dv(F,z)), X), f, m, m)
        EvaluateDoubleJacobian(X::AbstractVector{<:SymbolicScalar}) = _GetDoubleJacPass(F, X)
    end
end
function GetDoubleJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), f::Int=length(F(rand(m))), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :doublejacobian; kwargs...)   catch E;  
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff due to $E.")
        nothing
    end
    if isnothing(M)
        GetDoubleJac(Val(:ForwardDiff), F, m, f, args...)
    else M end
end
GetDoubleJac(ADmode::Val; Kwargs...) = (Dv=_GetDoubleJac(ADmode; Kwargs...);    EvaluateDoubleJacobian(F::Function, X, args...; kwargs...) = Dv(F, X, args...; kwargs...))
GetDoubleJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvalddF(F)

# For emergencies: needs an extra evaluation of function to determine length(Func(p))
function _GetDoubleJac(ADmode::Val; kwargs...)
    Dv = _GetJac(ADmode; kwargs...)
    Functor(Func::Function, X) = reshape(Dv(vec∘(z->Dv(Func,z)), X), length(Func(X)), length(X), length(X))
end



# Fall back to ForwardDiff as standard
_GetDeriv(ADmode::Val{true}; kwargs...) = _GetDeriv(Val(:ForwardDiff); kwargs...)
_GetGrad(ADmode::Val{true}; kwargs...) = _GetGrad(Val(:ForwardDiff); kwargs...)
_GetJac(ADmode::Val{true}; kwargs...) = _GetJac(Val(:ForwardDiff); kwargs...)
_GetHess(ADmode::Val{true}; kwargs...) = _GetHess(Val(:ForwardDiff); kwargs...)
_GetMatrixJac(ADmode::Val{true}; kwargs...) = _GetMatrixJac(Val(:ForwardDiff); kwargs...)
_GetDoubleJac(ADmode::Val{true}; kwargs...) = _GetDoubleJac(Val(:ForwardDiff); kwargs...)

# User has passed either SymbolicScalar or Vector{SymbolicScalar} to function, try to perfom symbolic passthrough
_GetDerivPass(F::Function, X) = SymbolicPassthrough(F(X), X, :derivative)
_GetGradPass(F::Function, X) = SymbolicPassthrough(F(X), X, :gradient)
_GetJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :jacobian)
_GetHessPass(F::Function, X) = SymbolicPassthrough(F(X), X, :hessian)
_GetMatrixJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :matrixjacobian)
_GetDoubleJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :doublejacobian)


## Implement this for new backends
_GetDeriv(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.derivative
_GetGrad(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.gradient
_GetJac(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.jacobian
_GetHess(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.hessian
##

# Error messages for unloaded backends
_GetDeriv(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded.")
_GetGrad(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded.")
_GetJac(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded.")
_GetHess(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded.")
# _GetMatrixJac and _GetDoubleJac already implemented in terms of _GetJac


GetGrad!(ADmode::Symbol, args...; kwargs...) = GetGrad!(Val(ADmode), args...; kwargs...)
GetJac!(ADmode::Symbol, args...; kwargs...) = GetJac!(Val(ADmode), args...; kwargs...)
GetHess!(ADmode::Symbol, args...; kwargs...) = GetHess!(Val(ADmode), args...; kwargs...)
GetMatrixJac!(ADmode::Symbol, args...; kwargs...) = GetMatrixJac!(Val(ADmode), args...; kwargs...)

# No Passthrough for these
GetGrad!(ADmode::Val, args...; kwargs...) = _GetGrad!(ADmode, args...; kwargs...)
GetJac!(ADmode::Val, args...; kwargs...) = _GetJac!(ADmode, args...; kwargs...)
GetHess!(ADmode::Val, args...; kwargs...) = _GetHess!(ADmode, args...; kwargs...)
GetMatrixJac!(ADmode::Val, args...; kwargs...) = _GetMatrixJac!(ADmode, args...; kwargs...)


# ForwardDiff as standard
GetGrad!(F::Function, args...; kwargs...) = GetGrad!(Val(:ForwardDiff), F, args...; kwargs...)
GetJac!(F::Function, args...; kwargs...) = GetJac!(Val(:ForwardDiff), F, args...; kwargs...)
GetHess!(F::Function, args...; kwargs...) = GetHess!(Val(:ForwardDiff), F, args...; kwargs...)
GetMatrixJac!(F::Function, args...; kwargs...) = GetMatrixJac!(Val(:ForwardDiff), F, args...; kwargs...)


# Evaluation of differentation operations into pre-specified arrays for functions which are themselves out-of-place
"""
    GetGrad!(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes gradients in-place via a backend specified by `ADmode`.
The function returned by `GetGrad!` has argument structure `(Y::AbstractVector, X::AbstractVector)` where the gradient of `F` evaluated at `X` is saved into `Y`.

Example:
```julia
Gradient! = GetGrad!(Val(:ForwardDiff), x->x[1]^2 - x[2]^3)
Y = Vector{Float64}(undef, 2)
Gradient!(Y, rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetGrad!(ADmode::Val, F::Function; kwargs...)
    Dv = _GetGrad!(ADmode; kwargs...)
    EvaluateGradient!(Y::AbstractVector{<:Number}, X::AbstractVector{<:Number}, args...) = Dv(Y, F, X, args...)
    EvaluateGradient!(Y::AbstractVector{<:SymbolicScalar}, X::AbstractVector{<:SymbolicScalar}) = _GetGradPass!(Y, F, X)
end

"""
    GetJac!(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes Jacobians in-place via a backend specified by `ADmode`.
The function returned by `GetJac!` has argument structure `(Y::AbstractMatrix, X::AbstractVector)` where the Jacobian of `F` evaluated at `X` is saved into `Y`.

Example:
```julia
Jacobian! = GetJac!(Val(:ForwardDiff), x->[x[1]^2, -x[2]^3, x[1]*x[2]])
Y = Matrix{Float64}(undef, 3, 2)
Jacobian!(Y, rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetJac!(ADmode::Val, F::Function; kwargs...)
    Dv = _GetJac!(ADmode; kwargs...)
    EvaluateJacobian!(Y::AbstractMatrix{<:Number}, X::AbstractVector{<:Number}, args...) = Dv(Y, F, X, args...)
    EvaluateJacobian!(Y::AbstractMatrix{<:SymbolicScalar}, X::AbstractVector{<:SymbolicScalar}) = _GetJacPass!(Y, F, X)
end

"""
    GetHess!(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes Hessians in-place via a backend specified by `ADmode`.
The function returned by `GetHess!` has argument structure `(Y::AbstractMatrix, X::AbstractVector)` where the Hessian of `F` evaluated at `X` is saved into `Y`.

Example:
```julia
Hessian! = GetHess!(Val(:ForwardDiff), x->x[1]^2 -x[2]^3 + x[1]*x[2])
Y = Matrix{Float64}(undef, 2, 2)
Hessian!(Y, rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetHess!(ADmode::Val, F::Function; kwargs...)
    Dv = _GetHess!(ADmode; kwargs...)
    EvaluateHess!(Y::AbstractMatrix{<:Number}, X::AbstractVector{<:Number}, args...) = Dv(Y, F, X, args...)
    EvaluateHess!(Y::AbstractMatrix{<:SymbolicScalar}, X::AbstractVector{<:SymbolicScalar}) = _GetHessPass!(Y, F, X)
end

"""
    GetMatrixJac!(ADmode::Val, F::Function; kwargs...) -> Function
Returns a function which computes Jacobians in-place for array-valued functions via a backend specified by `ADmode`.
The function returned by `GetMatrixJac!` has argument structure `(Y::AbstractArray, X::AbstractVector)` where the Jacobian of `F` evaluated at `X` is saved into `Y`.

Example:
```julia
Jacobian! = GetMatrixJac!(Val(:ForwardDiff), x->[x[1]^2 x[2]^3; x[1]*x[2] 2])
Y = Array{Float64}(undef, 2, 2, 2)
Jacobian!(Y, rand(2))
```
For available backends, see `diff_backends()`.
"""
function GetMatrixJac!(ADmode::Val, F::Function; kwargs...)
    Dv = _GetMatrixJac!(ADmode; kwargs...)
    EvaluateMatrixJacobian(Y::AbstractArray{<:Number}, X::AbstractVector{<:Number}, args...) = Dv(Y, F, X, args...)
    EvaluateMatrixJacobian(Y::AbstractArray{<:SymbolicScalar}, X::AbstractVector{<:SymbolicScalar}) = _GetMatrixJacPass!(Y, F, X)
end

# Need to extend this to functions F which are themselves also in-place
_GetGradPass!(Y, F::Function, X) = copyto!(Y, SymbolicPassthrough(F(X), X, :gradient))
_GetJacPass!(Y, F::Function, X) = copyto!(Y, SymbolicPassthrough(F(X), X, :jacobian))
_GetHessPass!(Y, F::Function, X) = copyto!(Y, SymbolicPassthrough(F(X), X, :hessian))
_GetMatrixJacPass!(Y, F::Function, X) = copyto!(Y, SymbolicPassthrough(F(X), X, :matrixjacobian))


_GetGrad!(ADmode::Symbol; kwargs...) = _GetGrad!(Val(ADmode); kwargs...)
_GetJac!(ADmode::Symbol; kwargs...) = _GetJac!(Val(ADmode); kwargs...)
_GetHess!(ADmode::Symbol; kwargs...) = _GetHess!(Val(ADmode); kwargs...)
_GetMatrixJac!(ADmode::Symbol; kwargs...) = _GetMatrixJac!(Val(ADmode); kwargs...)


_GetGrad!(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded. See currently loaded backends with diff_backends().")
_GetJac!(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded. See currently loaded backends with diff_backends().")
_GetHess!(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded. See currently loaded backends with diff_backends().")
_GetMatrixJac!(ADmode::Val{T}; kwargs...) where T = throw("Backend $T does not exist or is not currently loaded. See currently loaded backends with diff_backends().")


# Fall back to ForwardDiff as standard
_GetGrad!(ADmode::Val{true}; kwargs...) = _GetGrad!(Val(:ForwardDiff); kwargs...)
_GetJac!(ADmode::Val{true}; kwargs...) = _GetJac!(Val(:ForwardDiff); kwargs...)
_GetHess!(ADmode::Val{true}; kwargs...) = _GetHess!(Val(:ForwardDiff); kwargs...)
_GetMatrixJac!(ADmode::Val{true}; kwargs...) = _GetMatrixJac!(Val(:ForwardDiff); kwargs...)


## Implement this for new backends
_GetGrad!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.gradient!
_GetJac!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.jacobian!
_GetHess!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ForwardDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array
##



_ConsistencyCheck(Fexpr, var, deriv::Symbol) = _ConsistencyCheck(Fexpr, var, Val(deriv))
_ConsistencyCheck(Fexpr::AbstractVector{<:SymbolicScalar}, var::AbstractVector{<:SymbolicScalar}, ::Union{Val{:jacobian},Val{:doublejacobian}}) = nothing
_ConsistencyCheck(Fexpr::SymbolicScalar, var::AbstractVector{<:SymbolicScalar}, ::Union{Val{:gradient},Val{:hessian}}) = nothing
_ConsistencyCheck(Fexpr::SymbolicScalar, var::SymbolicScalar, ::Val{:derivative}) = nothing
_ConsistencyCheck(Fexpr::AbstractArray{<:SymbolicScalar}, var::Union{<:SymbolicScalar,<:AbstractVector{<:SymbolicScalar}}, ::Val{:matrixjacobian}) = nothing

const DerivTypes = [:derivative, :gradient, :jacobian, :hessian, :doublejacobian, :matrixjacobian]
function _ConsistencyCheck(Fexpr, var, deriv::Val{T}) where T
    if T ∉ DerivTypes
        throw("Invalid deriv type: $T.")
    else
        throw("Got :$T but Fexpr=$(typeof(Fexpr)) and arg=$(typeof(var)).")
    end
end

"""
Executes symbolic derivative as specified by `deriv::Symbol`.
"""
function SymbolicPassthrough(Fexpr::Union{<:AbstractArray{<:SymbolicScalar},<:SymbolicScalar}, var::Union{<:AbstractVector{<:SymbolicScalar},<:SymbolicScalar}, deriv::Symbol=:jacobian; simplify::Bool=true)
    _ConsistencyCheck(Fexpr, var, deriv)

    SymbolicDoubleJacobian(V::AbstractVector{<:SymbolicScalar}, z::SymbolicScalar; simplify::Bool=true) = SymbolicDoubleJacobian(V, [z]; simplify=simplify)
    SymbolicDoubleJacobian(V::AbstractVector{<:SymbolicScalar}, z::AbstractVector{<:SymbolicScalar}; simplify::Bool=true) = SymbolicMatrixJacobian(Symbolics.jacobian(V,z),z; simplify=simplify)
    SymbolicMatrixJacobian(M::AbstractArray{<:SymbolicScalar}, z::SymbolicScalar; simplify::Bool=true) = SymbolicMatrixJacobian(M, [z]; simplify=simplify)
    function SymbolicMatrixJacobian(M::AbstractArray{<:SymbolicScalar}, z::AbstractVector{<:SymbolicScalar}; simplify::Bool=true)
        reshape(Symbolics.jacobian(vec(M), z; simplify=simplify), size(M)..., length(z))
    end

    if deriv == :doublejacobian
        SymbolicDoubleJacobian(Fexpr, var; simplify=simplify)
    elseif deriv == :matrixjacobian
        SymbolicMatrixJacobian(Fexpr, var; simplify=simplify)
    else
        (@eval Symbolics.$deriv)(Fexpr, var; simplify=simplify)
    end
end
"""
    GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, inplace::Bool=false, parallel::Bool=false)
Computes symbolic derivatives, including `:jacobian`, `:gradient`, `:hessian` and `:derivative` which are specified via `deriv`.
Special care has to be taken that the correct `inputdim` is specified! Silent errors may occur otherwise.
"""
function GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, kwargs...)
    @assert deriv ∈ [:derivative, :gradient, :jacobian, :hessian, :doublejacobian, :matrixjacobian]
    @variables x X[1:inputdim]
    var = inputdim > 1 ? X : (try F(rand()); x catch; X end)
    Fexpr = KillAfter(F, var; timeout=timeout)
    # Warning already thrown in KillAfter
    isnothing(Fexpr) && return nothing
    GetSymbolicDerivative(Fexpr, var, deriv; kwargs...)
end
GetSymbolicDerivative(F::Function, deriv::Symbol; kwargs...) = GetSymbolicDerivative(F, GetArgLength(F), deriv; kwargs...)

function GetSymbolicDerivative(Fexpr::Union{<:AbstractArray{<:SymbolicScalar},<:SymbolicScalar}, var::Union{<:AbstractVector{<:SymbolicScalar},<:SymbolicScalar}, deriv::Symbol=:jacobian; simplify::Bool=true, inplace::Bool=false, parallel::Bool=false, kwargs...)
    derivative = SymbolicPassthrough(Fexpr, var, deriv; simplify=simplify)
    Builder(derivative, var; parallel=parallel, inplace=inplace, kwargs...)
end

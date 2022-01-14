var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DerivableFunctionsBase","category":"page"},{"location":"#DerivableFunctionsBase","page":"Home","title":"DerivableFunctionsBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is the documentation for DerivableFunctionsBase.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package provides the base functionality of DerivableFunctions.jl without loading all the backends, i.e. only for ForwardDiff.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For in-depth examples and explanations, please see the Documentation of main package DerivableFunctions.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DerivableFunctionsBase]","category":"page"},{"location":"#DerivableFunctionsBase.DerivableFunction","page":"Home","title":"DerivableFunctionsBase.DerivableFunction","text":"DerivableFunction(F::Function; ADmode::Union{Val,Symbol}=Val(:Symbolic))\nDerivableFunction(F::Function, testinput::Union{Number,AbstractVector{<:Number}}; ADmode::Union{Val,Symbol}=Val(:Symbolic))\nDerivableFunction(F::Function, dF::Function; ADmode::Union{Val,Symbol}=Val(:Symbolic))\nDerivableFunction(F::Function, dF::Function, ddF::Function)\n\nStores derivatives of a given function (as well as input-output dimensions) for potentially faster computations when derivatives are known.\n\n\n\n\n\n","category":"type"},{"location":"#DerivableFunctionsBase.Builder-Tuple{Union{Symbolics.Num, AbstractArray{var\"#s103\", N} where {var\"#s103\"<:Symbolics.Num, N}}, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.Builder","text":"Builder(Fexpr::Union{<:AbstractVector{<:Num},<:Num}, args...; inplace::Bool=false, parallel::Bool=false, kwargs...)\n\nBuilds RuntimeGeneratedFunctions from expressions via build_function().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetArgLength-Tuple{Function}","page":"Home","title":"DerivableFunctionsBase.GetArgLength","text":"GetArgLength(F::Function; max::Int=100) -> Int\n\nAttempts to determine input structure of F, i.e. whether it accepts Numbers or AbstractVectors and of what length. This is achieved by successively evaluating the function on rand(i) until the evaluation no longer throws errors. As a result, GetArgLength will be unable to determine the correct input structure if F errors on rand(i).\n\nnote: Note\nDoes NOT discriminate between Real and Vector{Real} of length one, i.e. Real↦+1. To disciminate between these two options, use _GetArgLength() instead.\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetDeriv-Tuple{Val, Function, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.GetDeriv","text":"GetDeriv(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the scalar derivative of F out-of-place via a backend specified by ADmode.\n\nExample:\n\nDerivative = GetDeriv(Val(:ForwardDiff), x->exp(-x^2))\nDerivative(5.0)\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetDoubleJac","page":"Home","title":"DerivableFunctionsBase.GetDoubleJac","text":"GetDoubleJac(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the Jacobian of the Jacobian for a vector-valued function F out-of-place via a backend specified by ADmode.\n\nExample:\n\nDoubleJacobian = GetDoubleJac(Val(:ForwardDiff), x->[x[1]^2, x[1]*x[2]^3])\nDoubleJacobian(rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"function"},{"location":"#DerivableFunctionsBase.GetGrad!-Tuple{Val, Function}","page":"Home","title":"DerivableFunctionsBase.GetGrad!","text":"GetGrad!(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes gradients in-place via a backend specified by ADmode. The function returned by GetGrad! has argument structure (Y::AbstractVector, X::AbstractVector) where the gradient of F evaluated at X is saved into Y.\n\nExample:\n\nGradient! = GetGrad!(Val(:ForwardDiff), x->x[1]^2 - x[2]^3)\nY = Vector{Float64}(undef, 2)\nGradient!(Y, rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetGrad-Tuple{Val, Function, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.GetGrad","text":"GetGrad(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the gradient of F out-of-place via a backend specified by ADmode.\n\nExample:\n\nGradient = GetGrad(Val(:ForwardDiff), x->x[1]^2 - x[2]^3)\nGradient(rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetHess!-Tuple{Val, Function}","page":"Home","title":"DerivableFunctionsBase.GetHess!","text":"GetHess!(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes Hessians in-place via a backend specified by ADmode. The function returned by GetHess! has argument structure (Y::AbstractMatrix, X::AbstractVector) where the Hessian of F evaluated at X is saved into Y.\n\nExample:\n\nHessian! = GetHess!(Val(:ForwardDiff), x->x[1]^2 -x[2]^3 + x[1]*x[2])\nY = Matrix{Float64}(undef, 2, 2)\nHessian!(Y, rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetHess-Tuple{Val, Function, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.GetHess","text":"GetHess(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the Hessian of F out-of-place via a backend specified by ADmode.\n\nExample:\n\nHessian = GetHess(Val(:ForwardDiff), x->x[1]^2 -x[2]^3 + x[1]*x[2])\nHessian(rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetJac!-Tuple{Val, Function}","page":"Home","title":"DerivableFunctionsBase.GetJac!","text":"GetJac!(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes Jacobians in-place via a backend specified by ADmode. The function returned by GetJac! has argument structure (Y::AbstractMatrix, X::AbstractVector) where the Jacobian of F evaluated at X is saved into Y.\n\nExample:\n\nJacobian! = GetJac!(Val(:ForwardDiff), x->[x[1]^2, -x[2]^3, x[1]*x[2]])\nY = Matrix{Float64}(undef, 3, 2)\nJacobian!(Y, rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetJac-Tuple{Val, Function, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.GetJac","text":"GetJac(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the Jacobian of F out-of-place via a backend specified by ADmode.\n\nExample:\n\nJacobian = GetJac(Val(:ForwardDiff), x->[x[1]^2, -x[2]^3, x[1]*x[2]])\nJacobian(rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetMatrixJac","page":"Home","title":"DerivableFunctionsBase.GetMatrixJac","text":"GetMatrixJac(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes the Jacobian of an array-valued function F out-of-place via a backend specified by ADmode.\n\nExample:\n\nJacobian = GetMatrixJac(Val(:ForwardDiff), x->[x[1]^2 x[2]^3; x[1]*x[2] 2])\nJacobian(rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"function"},{"location":"#DerivableFunctionsBase.GetMatrixJac!-Tuple{Val, Function}","page":"Home","title":"DerivableFunctionsBase.GetMatrixJac!","text":"GetMatrixJac!(ADmode::Val, F::Function; kwargs...) -> Function\n\nReturns a function which computes Jacobians in-place for array-valued functions via a backend specified by ADmode. The function returned by GetMatrixJac! has argument structure (Y::AbstractArray, X::AbstractVector) where the Jacobian of F evaluated at X is saved into Y.\n\nExample:\n\nJacobian! = GetMatrixJac!(Val(:ForwardDiff), x->[x[1]^2 x[2]^3; x[1]*x[2] 2])\nY = Array{Float64}(undef, 2, 2, 2)\nJacobian!(Y, rand(2))\n\nFor available backends, see diff_backends().\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetOutLength-Tuple{Function, Union{Number, AbstractVector{var\"#s107\"} where var\"#s107\"<:Number}}","page":"Home","title":"DerivableFunctionsBase.GetOutLength","text":"GetOutLength(F::Function, input::Union{Number,AbstractVector{<:Number}})\n\nReturns output dimensions of given F. If it outputs arrays of more than one dimension, a tuple is returned. This can also be used to determine the approximate size of the input for mutating F which accept 2 arguments.\n\nnote: Note\nDiscriminates between Real and Vector{Real} of length one, i.e.: Real↦-1 and x::AbstractVector{<:Real}↦length(x).\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.GetSymbolicDerivative","page":"Home","title":"DerivableFunctionsBase.GetSymbolicDerivative","text":"GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, inplace::Bool=false, parallel::Bool=false)\n\nComputes symbolic derivatives, including :jacobian, :gradient, :hessian and :derivative which are specified via deriv. Special care has to be taken that the correct inputdim is specified! Silent errors may occur otherwise.\n\n\n\n\n\n","category":"function"},{"location":"#DerivableFunctionsBase.KillAfter-Tuple{Function, Vararg{Any, N} where N}","page":"Home","title":"DerivableFunctionsBase.KillAfter","text":"KillAfter(F::Function, args...; timeout::Real=5, verbose::Bool=false, kwargs...)\n\nTries to evaluate a given function F before a set timeout limit is reached and interrupts the evaluation and returns nothing if necessary. NOTE: The given function is evaluated via F(args...; kwargs...).\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.MaximalNumberOfArguments-Tuple{Function}","page":"Home","title":"DerivableFunctionsBase.MaximalNumberOfArguments","text":"MaximalNumberOfArguments(F::Function) -> Int\n\nInfers argument structure of given function, i.e. whether it is of the form F(x) or F(x,y) or F(x,y,z) etc. and returns maximal number of accepted arguments of all overloads of F as integer.\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.SymbolicPassthrough","page":"Home","title":"DerivableFunctionsBase.SymbolicPassthrough","text":"Executes symbolic derivative as specified by deriv::Symbol.\n\n\n\n\n\n","category":"function"},{"location":"#DerivableFunctionsBase._GetArgLength-Tuple{Function}","page":"Home","title":"DerivableFunctionsBase._GetArgLength","text":"_GetArgLength(F::Function; max::Int=100) -> Int\n\nnote: Note\nDiscriminates between Real and Vector{Real} of length one, i.e.: Real↦-1 and x::AbstractVector{<:Real}↦length(x).\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.diff_backends-Tuple{}","page":"Home","title":"DerivableFunctionsBase.diff_backends","text":"Shows the differentation backends available for use with DerivableFunctions.jl.\n\n\n\n\n\n","category":"method"},{"location":"#DerivableFunctionsBase.suff-Tuple{BigFloat}","page":"Home","title":"DerivableFunctionsBase.suff","text":"suff(x) -> Type\n\nIf x stores BigFloats, suff returns BigFloat, else suff returns Float64.\n\n\n\n\n\n","category":"method"}]
}

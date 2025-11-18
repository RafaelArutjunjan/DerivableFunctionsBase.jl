module DerivableFunctionsDifferentiationInterfaceExt

using DerivableFunctionsBase, DifferentiationInterface, ADTypes

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetDoubleJac, _GetMatrixJac
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!


_GetDeriv(ADmode::ADTypes.AbstractADType; kwargs...) = (Func::Function,p;Kwargs...) -> DifferentiationInterface.derivative(Func, ADmode, p; kwargs...)
_GetGrad(ADmode::ADTypes.AbstractADType; kwargs...) = (Func::Function,p;Kwargs...) -> DifferentiationInterface.gradient(Func, ADmode, p; kwargs...)
_GetJac(ADmode::ADTypes.AbstractADType; kwargs...) = (Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian(Func, ADmode, p; kwargs...)
_GetHess(ADmode::ADTypes.AbstractADType; kwargs...) = (Func::Function,p;Kwargs...) -> DifferentiationInterface.hessian(Func, ADmode, p; kwargs...)
_GetDoubleJac(ADmode::ADTypes.AbstractADType; kwargs...) = (Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian(Func, ADmode, p; kwargs...)

_GetGrad!(ADmode::ADTypes.AbstractADType; kwargs...) = (G,Func::Function,p;Kwargs...) -> DifferentiationInterface.gradient!(Func, G, ADmode, p; kwargs...)
_GetJac!(ADmode::ADTypes.AbstractADType; kwargs...) = (J,Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian!(Func, J, ADmode, p; kwargs...)
_GetHess!(ADmode::ADTypes.AbstractADType; kwargs...) = (H,Func::Function,p;Kwargs...) -> DifferentiationInterface.hessian!(Func, H, ADmode, p; kwargs...)
_GetDoubleJac!(ADmode::ADTypes.AbstractADType; kwargs...) = (J,Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian!(Func, J, ADmode, p; kwargs...)

_GetGrad!(ADmode::ADTypes.AbstractADType, Prep::DifferentiationInterface.Prep; kwargs...) = (G,Func::Function,p;Kwargs...) -> DifferentiationInterface.gradient!(Func, G, Prep, ADmode, p; kwargs...)
_GetJac!(ADmode::ADTypes.AbstractADType, Prep::DifferentiationInterface.Prep; kwargs...) = (J,Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian!(Func, J, Prep, ADmode, p; kwargs...)
_GetHess!(ADmode::ADTypes.AbstractADType, Prep::DifferentiationInterface.Prep; kwargs...) = (H,Func::Function,p;Kwargs...) -> DifferentiationInterface.hessian!(Func, H, Prep, ADmode, p; kwargs...)
_GetDoubleJac!(ADmode::ADTypes.AbstractADType, Prep::DifferentiationInterface.Prep; kwargs...) = (J,Func::Function,p;Kwargs...) -> DifferentiationInterface.jacobian!(Func, J, Prep, ADmode, p; kwargs...)


_GetDeriv(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetDeriv(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetGrad(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetGrad(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetJac(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetJac(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetHess(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetHess(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetDoubleJac(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetDoubleJac(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")

_GetGrad!(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetGrad!(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetJac!(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetJac!(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetHess!(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetHess!(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetDoubleJac!(ADmode::Val{T}; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetDoubleJac!(T; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")

_GetGrad!(ADmode::Val{T}, Prep::DifferentiationInterface.Prep; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetGrad!(T, Prep; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetJac!(ADmode::Val{T}, Prep::DifferentiationInterface.Prep; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetJac!(T, Prep; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetHess!(ADmode::Val{T}, Prep::DifferentiationInterface.Prep; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetHess!(T, Prep; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")
_GetDoubleJac!(ADmode::Val{T}, Prep::DifferentiationInterface.Prep; kwargs...) where T = T isa ADTypes.AbstractADType ? _GetDoubleJac!(T, Prep; kwargs...) : throw("Do not know how to handle $T. Backend possibly not loaded yet.")


import DerivableFunctionsBase: _add_backend
__init__() = _add_backend(:DifferentiationInterface)

end # module
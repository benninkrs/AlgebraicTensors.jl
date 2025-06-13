"""
	AlgebraicTensors (module)

Implements tensors of arbitrary dimensionality and size as algebraic objects.
"""

#=
8/20/2023
	This is a major revision to (1) make getindex type stable and (2) use the new TensorOperation.jl API.
	I am going to return to the original idea parameterizing the type with tuples for the ordered left and
	right spaces. Hopefully compiler improvements have made this inferrable now.
	This should make all typed functions fast, but may result in excessive code generation.
=#

#=
- In A+B and A-B, the order of spaces is that of A
- In A*B, the left spaces of A precede the uncontracted left spaces of B,
	followed by the uncontracted right spaces of A, then right spaces of B.
	When applying a tensor operator to a tensor vector, the order of spaces
	in the vector will be changed.  This is sort of undesirable ... ?
=#


#=
COMPLETED:
Constructors			Done, performant
getindex					Done, performant
setindex!				Done, performant
==							Done, performant
+,-						Done, performant (but can be better)
adjoint					Done, performant
transpose				Done, performant
partial transpose		Done, performant
Tensor*Tensor			Done, performant
trace						Done, performant
partial trace			Done, performant
analytic funcs			Done, performant
det						Done, performant
opnorm					Done, performant
eigvals					Done, performant
svdvals					Done, performant
eig						Done, performant
svd						Dome, performant


TODO:
- Put spaces in ascending order for:  *, partial transpose, + and -?
- Implement tensor contraction using generated functions instead of TensorOperatios?
- Implement IndexLinear versions for +,-,== (will be faster)
- Tensor*Array?
- broadcasting
- lazy tensor products
- test performance in non-inferrable cases
=#


module AlgebraicTensors

export AbstractTensor, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
export tr, marginal, det, opnorm, eigvals, eigvecs, svdvals, svd, opnorm

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: Index2Tuple, tensortrace, tensorcontract, tensoradd, tensorscalar, tensorproduct

using Base.Broadcast: Broadcasted, BroadcastStyle
import Base: display, show
import Base: ndims, length, size, axes, similar, tail
import Base: reshape, permutedims, adjoint, transpose, Matrix, == #, isapprox
import Base: similar
import Base: getindex, setindex!
using Base: tail
import Base: (+), (-), (*), (/), (^)
import Base: inv, exp, log, sin, cos, tan, sinh, cosh, tanh
import LinearAlgebra: tr, det, eigvals, svdvals, opnorm, eigvecs, svd



#------------------------------------
# Definitions

const SpacesInt = UInt128  		# An integer treated as a bit set for vector spaces
const MaxSpace = 	128				# number of bits in SpacesInt (maximum space index)
const Spaces{N} = Tuple{Vararg{Integer,N}}
const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}
const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}
const SupportedArray{T,N} = DenseArray{T,N}		# can change this later.  Maybe to StridedArray?


# Abstract tensor with left spaces LS, right spaces RS, and element type T
abstract type AbstractTensor{LS, RS, T, N} <: AbstractArray{T,N} end


lspaces(::AbstractTensor{LS,RS}) where {LS, RS} = LS
rspaces(::AbstractTensor{LS,RS}) where {LS,RS} = RS
spaces(M::AbstractTensor) = (lspaces(M)..., rspaces(M)...)

nlspaces(M) = length(lspaces(M))
nrspaces(M) = length(rspaces(M))

#-------------------------------------
# Internal utility functions

# return an Int whose set bits correspond to the left or right spaces
lspaces_int(M::AbstractTensor) = binteger(SpacesInt, Val(lspaces(M)))
rspaces_int(M::AbstractTensor) = binteger(SpacesInt, Val(rspaces(M)))

# the array dimensions corresponding to the sorted left and right spaces 
lspace_srt_dims(M::AbstractTensor) = sortperm(lspaces(M))
rspace_srt_dims(M::AbstractTensor) = nlspaces(M) .+ sortperm(rspaces(M))

#-----------------------------------------

include("tensor.jl")
include("tensorproduct.jl")

end
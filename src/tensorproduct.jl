#  Tensorpoduct.jl
#
#=
A factorized representation of tensors is going to be tricky to implement.

=#


export TensorProduct, ⊗


⊗(x::Number, y::Number) = x*y
⊗(T::Tensor, x::Number) = T*x
⊗(x::Number, T::Tensor) = x*T

⊗(a, b, args...) = ⊗((a⊗b), args...)


#=
Initially this was formulated so that each factor had distinct spaces.
However, this would rule out useful constructions such as T[i,j,k] = A[i,k]*B[j,k].

Allowing factors to have spaces in common introduces many complications.
For example, the output spaces cannot simply be the spaces of the factor
in order of appaearnce.
Also, it requires a more general type of tensor (and tensor contraction!), e.g.
tensor netwoorks that are hypergraphs.  Can TensorOperations handle these?
=#
"""
	TensorProduct(A,B,C...)

Construct a tensor whose elements are a product of corresponding elements from A,B,C...
This is more general than an outer product in that A,B,C,... may have spaces in common.
The spaces of the resulting tensor are sorted.
`TensorProduct`s are formed by multiplying tensors together via ⊗.
"""
struct TensorProduct{LS, RS, T, N, TT<:Tuple{Vararg{Tensor}}} <: AbstractTensor{LS,RS,T,N}
	factors::TT

	function TensorProduct(factors::Tensor...)
		LS = SpacesInt(0)
		RS = SpacesInt(0)

		for t in factors
			ls = lspaces_int(t)
			rs = rspaces_int(t)

			# Can have spaces in common, but axes must match.
			# We need a way to track axes of each space

			
			# ls & LS == SpacesInt(0) || error("Factors must have distinct left spaces")
			# rs & RS == SpacesInt(0) || error("Factors must have distinct right spaces")

			# 

			LS |= ls
			RS |= rs
		end

		lspaces_ = tcat((lspaces(t) for t in factors)...)
		rspaces_ = tcat((rspaces(t) for t in factors)...)

		T = promote_type((eltype(t) for t in factors)...)
		N = length(lspaces_) + length(rspaces_)
		return new{lspaces_, rspaces_, T, N, typeof(factors)}(factors)	
	end
end

ndims(TP::TensorProduct) = nlspaces(TP) + nrspaces(TP)



# Given the spaces of each factor, return a tuple t such that
# t[i] gives the factor and dimension of the ith sorted space
function factors_dims(::Val{spaces}) where {spaces}
	allspaces = 
	nf = length(spaces)

end

#---------------------------------
# Is this the simplest way of doing this?

# Find the factor and dimension associated with each lspace
function lspace_to_factor_dim(TP::TensorProduct, spc)
	lengths = ntuple(i->nlspaces(TP.factors[i]), nfactors(TP))
	cumlengths = (0, cumsum(lengths)...)
	allspaces = tcat(ntuple(i->lspaces(TP.factors[i]), nfactors(TP))...)
	spc_idx = findin(allspaces, spc)

	# Given the index of spc in allspaces, combined with the 
	return find_factor_dim(cumlengths, spc_idx)
end


# Helper function for (l,r)space_to_factor_dim
function find_factor_dim(cumlengths, i)
	factor = findfirst(c -> c>=i, cumlengths) - 1
	dim = i - cumlengths[factor]
	return (factor, dim)
end


#---------------------



#------
# All these need to be modified to handle overlapping spaces.
# Basically, we need to count each lspace and each rspace once.
# This seems to require a mapping from spaces to factors and dimensions
length(TP::TensorProduct) = prod(length(t.data) for t in TP.factors)


size(TP::TensorProduct) = tcat((size(t.data) for t in TP.factors)...)
size(TP::TensorProduct, i) = size(TP)[i]
lsize(TP::TensorProduct) = tcat((lsize(t) for t in TP.factors)...)
rsize(TP::TensorProduct) = tcat((rsize(t) for t in TP.factors)...)

axes(TP::TensorProduct) = tcat((axes(t.data) for t in TP.factors)...)
axes(TP::TensorProduct, i) = axes(TP)[i]
laxes(TP::TensorProduct) = tcat((laxes(t) for t in TP.factors)...)
raxes(M::TensorProduct) = tcat((raxes(t) for t in TP.factors)...)
#----

nfactors(TP::TensorProduct) = length(TP.factors)

display(M::TensorProduct) = show(M)

function show(io::IO, M::TensorProduct)
	# print as usual
	ls = join(map(string, lsize(M)), '×')
	rs = join(map(string, rsize(M)), '×')
	print(io, '(', ls, ")×(", rs, ") ")
	print(io, "TensorProduct{", lspaces(M), "←→", rspaces(M), ", ",
					eltype(M), "} with ", length(M.factors), " factors")

	if !isempty(M.factors)
		println(io)
		for (i,t) in enumerate(M.factors)
			println(io, "Factor ", i, ":")
			show(io, t)
			println(io)
		end
	end
end


# Return the dimensions associated with a particular factor.
function factor_dims(TP::TensorProduct, i)
	lsp = lspaces(TP)
	rsp = rspaces(TP)

	ldims = findin(lspaces(TP.factors[i]), lsp)
	rdims = findin(rspaces(TP.factors[i]), rsp)

	dims = (ldims..., rdims...)
	println("Factor ", i, ": dims = ", dims)
end


function getindex(TP::TensorProduct, I...)
	# TODO: Convert I to one set of indices per dimension
	#			(e.g. CartesianIndex needs to be splatted)
	
	if length(I) > ndims(TP)
		# TODO:  Support trailing "1" indices
		error("Too many dimensions in indices")
	end

	if nfactors(TP) < 1 && length(I) > 0
		error("Cannot index a TensorProduct that has no factors")
	end

	# Amazingly, the type is inferred.
	# But are all the intermediaries stored simultaneously?
	vals = ntuple(i -> TP.factors[i][I[factor_dims(TP,i)]...], nfactors(TP))
	return ⊗(vals...)

end



# Return a tuple with the dimensions corresponding to each factor
function factor_dims_old(TP::TensorProduct)
	llengths = ((nlspaces(t) for t in TP.factors)...,)
	rlengths = ((nrspaces(t) for t in TP.factors)...,)
	lidx = 1 .+ (0, cumsum(llengths)...)
	ridx = lidx[end] .+ (0, cumsum(rlengths)...)
	dims = ntuple(i -> ((lidx[i]:lidx[i+1]-1)..., (ridx[i]:ridx[i+1]-1)...), length(llengths))
end



function getindex_old(TP::TensorProduct, I...)
	# convert I to one set of indices per dimension
	# check for extraneous dimensions?

	if nfactors(TP) < 1 && length(I) > 0
		error("Cannot index a TensorProduct that has no factors")
	end

	dims = factor_dims(TP)

	# println("I = ", I)
	# println("dims = ", dims)

	# Amazingly, the type is inferred.
	# But are all the intermediaries stored simultaneously?
	vals = ntuple(i -> TP.factors[i][I[dims[i]]...], nfactors(TP))
	return ⊗(vals...)

	# This doesn't always get inferred
	# v0 = TP.factors[1][I[dims[1]]...]
	# v = SuperTuples.static_iter_f_ix((i,x) -> x ⊗ TP.factors[i+1][I[dims[i+1]]...], v0, Val(nfactors(TP)-1))
	# return v
end



*(x::Number, TP::TensorProduct) = TensorProduct(x*TP.factors[1], tail(TP.factors)...)
*(TP::TensorProduct, x::Number) = TensorProduct(TP.factors[1]*x, tail(TP.factors)...)


# This is 2x slower than raw vector multiplication due to the permutedims.
# Same speed as *.
# TODO:  Move to tensorproduct.jl and have it produce a TensorProduct instead
"""
	⊗(A::Tensor, B::Tensor)

Tensor (outer product). The left (right) spaces of `A` must be distinct from the
left (right) spaces of `B`. The output spaces are
	lspaces(A⊗B) = (lspaces(A)..., lspaces(B)...)
	rspaces(A⊗B) = (rspaces(A)..., rspaces(B)...)

For some arguments the outer product could also be obtained using `*`, but in that
case the output spaces would be sorted. 
"""
function ⊗(A::Tensor, B::Tensor)
	LA = lspaces_int(A)
	RA = rspaces_int(A)
	LB = lspaces_int(B)
	RB = rspaces_int(B)

	LA & LB == SpacesInt(0) || error("A and B must not have any common left spaces")
	RA & RB == SpacesInt(0) || error("A and B must not have any common right spaces")

	lspaces_ = (lspaces(A)..., lspaces(B)...)
	rspaces_ = (rspaces(A)..., rspaces(B)...)

	data_m = vec(A.data) * transpose(vec(B.data))
	data_ = reshape(data_m, (size(A)..., size(B)...))
	perm = blockperm((nlspaces(A), nrspaces(A), nlspaces(B), nrspaces(B)), (1,3,2,4))
	return Tensor{lspaces_, rspaces_}(permutedims(data_, perm))
end


# This is slower for small matrices.
# export outprod
# function outprod(A::Tensor, B::Tensor)
# 	LA = lspaces_int(A)
# 	RA = rspaces_int(A)
# 	LB = lspaces_int(B)
# 	RB = rspaces_int(B)

# 	LA & LB == SpacesInt(0) || error("A and B must not have any common left spaces")
# 	RA & RB == SpacesInt(0) || error("A and B must not have any common right spaces")

# 	lspaces_ = (lspaces(A)..., lspaces(B)...)
# 	rspaces_ = (rspaces(A)..., rspaces(B)...)

# 	pc1 = blockperm((nlspaces(A), nrspaces(A), nlspaces(B), nrspaces(B)), (1,3))
# 	pc2 = blockperm((nlspaces(A), nrspaces(A), nlspaces(B), nrspaces(B)), (2,4))
# 	pA = (oneto(ndims(A)), ())
# 	pB = ((), oneto(ndims(B)))
# 	pC = (pc1, pc2)

# 	# println("pA = ", pA)
# 	# println("pB = ", pB)
# 	# println("pC = ", pC)
# 	data_ = tensorcontract(A.data, pA, false, B.data, pB, false, pC)
# 	return Tensor{lspaces_, rspaces_}(data_)
# end


# TODO:
# convert TensorProduct to Tensor
# 
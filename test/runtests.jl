using AlgebraicTensors
using BenchmarkTools
using LinearAlgebra: I, tr
using Test

# Times shown are for Julia 1.11.6 on my HP ZBook Firefly G10, i7-1360P.
# (But Pkg.test needs to be run with --check-bounds=auto, otherwise bounds will always be checked
# and results will be slower.)

A = Tensor{(1,2)}(reshape(1.0:24, (1,2,3,4)));
B = Tensor{1:2}(reshape(1.0:24, (3,4,1,2)));
C = Tensor{(1,2)}(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)));
R = Tensor{(1,2)}(rand(3,4,3,4));
R_ = Tensor{(1,2),(2,1)}(permutedims(R.data, (1,2,4,3)));

AA = Tensor{(4,3,2,1), (5,6,7)}(rand(4,3,2,1,5,6,7));
BB = Tensor{(5,6,7,8), (11,10,9)}(rand(5,6,7,5,4,3,2));
RR = Tensor{(1,2)}(rand(10,20,10,20));
RR_ = Tensor{(1,2),(2,1)}(rand(10,20,20,10));


@testset "construction" begin
    @info "Testing (re)construction"
    A_ = A(3,2);
    @test (lspaces(A_), rspaces(A_)) == ((3,2), (3,2))
    @btime A_ = ($A)(3,2);      		# 1.4μs (14 allocations)
    @btime A_ = ($A)((3,2));      	# 1.4μs (14 allocations)
    @btime A_ = ($A)((3,2),(3,2));	# 1.4μs (14 allocations)

    A_ = A((5,6),(8,7));
    @test (lspaces(A_), rspaces(A_)) == ((5,6), (8,7))
    @btime A_ = ($A)((5,6),(8,7));	# 1.4μs (14 allocations)
end


@testset "indexing" begin
    @info "testing getindex"
    T = Tensor{(5,2,3),(10,60)}(randn(2,3,4,5,6));
    S = T[:,2,:,4,:];
    @test (lspaces(S), rspaces(S)) == ((5,3), (60,))
    @btime ($T)[:,2,:,4,:];             # 52 ns (1 alloc)
end


@testset "equality" begin
    @info "Testing =="
    A_ = A((2,1));
    A__ = Tensor{(2,1)}(permutedims(A.data, (2,1,4,3)));
    AA_ = AA((4,3,2,1),(7,6,5));
    AA__ = Tensor{(1,2,3,4),(7,6,5)}(permutedims(AA.data, (4,3,2,1,7,6,5)));

    @test A == A
    @test A != A_
    @test A == A__
    @btime $A == $A;    	# 19 ns
    @btime $A == $A__;     # 28 ns (with @inbounds)

    @test AA == AA
    @test AA != AA_
    @test AA == AA__
    @btime $AA == $AA;      # 3.6 μs
    @btime $AA == $AA__;    # 3.9 μs
end


@testset "add" begin
    @info "testing +,-"
    X = Tensor{(3,5),(4,8)}(reshape(1.0:1.0:24, (1,2,3,4)));
    Y = permutedims(2*X, (2,1,4,3))((5,3),(8,4));
    @test X+X == Y;
    @test X+Y == 3*X;
    @btime $X+$X;   			# 84 ns (2 allocations)
    @btime $X+$Y;   			# 82 ns (2 allocations)

    XX = Tensor{(1,2,3,4), (7,6,5)}(rand(1,2,3,4,7,6,5));
    @btime $AA+$AA;  	  # 7.5 μs, 3 allocations
    @btime $AA+$XX;	     # 6.4 μs, 3 allocations
end


@testset "transpose" begin
    @info "Testing transpose"
    T = Tensor{(2,5,3),(1,5)}(randn(2,3,4,5,6));
    S = transpose(T, 5);
    @test (lspaces(S), rspaces(S)) == ((2,5,3), (1,5))
    @test size(S) == (2,6,4,5,3)
    @btime transpose($T, 5);			# 708 ns, 5 allocations

    S = transpose(T, (5,2));
    @test (lspaces(S), rspaces(S)) == ((5,3), (1,5,2))
    @test size(S) == (6,4,5,3,2)
    @btime transpose($T, (5,2));		# 623 ns, 5 allocations
end

@testset "mult" begin
    @info "Testing multiplication"
    @test A*B == C
    @test_throws DimensionMismatch A*B(2,1)
    C_ = A(9,5) * B(9,5)
    @test spaces(C_) == (9,5,9,5)
    @test C_== C(9,5)

    CC = AA*BB;
    @test (lspaces(CC), rspaces(CC)) == ((4,3,2,1,8), (11,10,9))

    @info "Benchmarking small multiplication."  	
    @btime $A*$B;				# 140 ns (5 allocations: 272 bytes)

    @info "Benchmarking large multiplication."		
    @btime $AA*$BB;			# 60 μs (9 allocations: 23 KiB)
end


@testset "outprod" begin
    @info "Testing outer product"
    P = Tensor{(1,2),(3,4,5)}(randn(2,3,4,5,6))
    Q = Tensor{(3,4),(7,)}(randn(4,3,2))

    PQ = P⊗Q
    sz = (2,3,4,3,4,5,6,2)
    @test spaces(PQ) == (1,2,3,4,3,4,5,7)
    @test size(PQ) == sz
    @test PQ.data == permutedims(reshape(vec(P.data) * vec(Q.data)', (size(P)..., size(Q)...)), (1, 2, 6, 7, 3, 4, 5, 8))

    @info "Benchmarking outer product"    		# 28 μs (11 allocations)
    @btime $P⊗$Q
end


@testset "trace" begin
    @info "Benchmarking trace"
    # tr(Matrix(R)) takes only 14 ns.
    @btime tr($R);          # 8 ns (0 allocations)
    @btime tr($R_);         # 25 ns (0 allocations)
    @btime tr($RR);         # 86 ns
    @btime tr($RR_);         # 127 ns

    @info "Benchmakring partial trace"	
    @btime tr($R, 2);            # 155 ns (4 allocations)
    @btime tr($R_, 2);           # 172 ns (4 allocations)
    @btime tr($R, Val((2,)));    # 53 ns (1 allocations)

    @btime tr($RR, 2);           # 1.2 μs (4 allocations)
    @btime tr($RR, Val((2,)));   # 1.0 μs (1 allocations)

    @info "Benchmarking marginal" 
    @btime marginal($R, 1);       # 156 ns (0 allocations)
end


@testset "eig" begin
    @info "Testing eig"
    e = eigvals(R);
    e_ = eigvals(R_)
    @test e == e_;

    V = eigvecs(R);
    V_ = eigvecs(R_);
    @test Matrix(V) == Matrix(V_)
    @test Matrix(V) == eigvecs(Matrix(R))
end


@testset "svd" begin
    @info "Testing svd"
    s = svdvals(R)
    s_ = svdvals(R_)
    @test s == s_ == svdvals(Matrix(R))

    (U,S,Vt) = svd(R);
    result = svd(Matrix(R));
    @test Matrix(U) == result.U
    @test Matrix(Vt) == result.Vt
    @test S == result.S

    (U_,S_,Vt_) = svd(R_);
    @test U ≈ U_
    @test S ≈ S_
    @test Vt ≈ Vt_
end


@testset "combination" begin
    # @info "Combination tests"
    @test size(A + B') == (1,2,3,4)
    @test R * inv(R) ≈ Tensor{1:2}(reshape(I(12), (3,4,3,4)))
end
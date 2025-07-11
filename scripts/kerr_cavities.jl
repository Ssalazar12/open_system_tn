using DrWatson
@quickactivate

using HDF5
using Printf
using SparseArrays
using LinearAlgebra
using Statistics
using Distributed

using QuantumToolbox
using ITensors, ITensorMPS
using Strided


N = 3 # local hilbert space dimension
L = 10 # size of the chain
ω = 2.5 # frequency of the harmonic oscillator
J = 2.0 # hopping
γ = 1.0 # damping rate
γ_list = γ.*ones(L)
U = 0.1 # Kerr non-linearity
F_list = [0.2, 1.5, 2.5]
maxt = 10.0 # maximum time
# Do an ensemble average over all trajectories to get the actual occupatinos
N_trajectories = 10
cutoff_exponent = -18 # -18 -20
cutoff = 10.0^(cutoff_exponent)
Delta_t = 0.2
time_list = 0.0:Delta_t:maxt
γ_list = γ.*ones(L)
evol_type = "TEBD" #"TE

# so there is no competition on multithreading
Strided.disable_threads()


# --------------------------------
# FUNCTIONS 
# --------------------------------

function ITensors.op(::OpName"Kerr", ::SiteType"Boson", d::Int)
    # Define a custom operator for the Kerr  using Quantum toolbox
    a = destroy(d) 
    kerr_mat = a'*a'*a*a
    
  return to_dense(kerr_mat.data)
end

function entangement_S(ψ, b)
    # b: index where we do the bipartition
    psi = orthogonalize(ψ, b)
    # do the SVM
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    # Geth the schmidt coefficients
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end


function ask_where(intervals)
    # decide where to project the staet when there is a jump, each element in intervals is associated to
    # a jump operator
    r2 = rand()
    dummy = 0
    j = 1
    found_site = -1
    while dummy == 0
        if r2<sum(intervals[1:j])
            dummy = 1
            found_site = j
        else 
            j = j+1
        end
    end
    return found_site
end

function build_TEBD(L, Ω, K ,J, Fd, Δτ, Nh, gammas::Vector{Float64})
    #  build site indices
    s = siteinds("Boson", L; dim=Nh, conserve_qns=false)
    # build the time evol gates for TEBD
    gates = ITensor[]
    # for the first site 
    s1 = s[1]
    s2 = s[2]
    h0 = -Ω*op("n",s1)*op("Id",s2) + 0.5*K*op("Kerr",s1)*op("Id",s2) + Fd*(op("adag",s1) + op("a",s1))*op("Id",s2)
    
    heff = - 0.5*im*gammas[1]*op("Id",s1)*op("n",s2)
    hj = h0 +  heff
    Gj = exp(-im * Δτ/2 * hj)
    push!(gates, Gj)

    for j in 2:(L)
        s1 = s[j]
        s2 = s[j-1] 

        h0 = -Ω*op("n",s1)*op("Id",s2) + 0.5*K*op("Kerr",s1)*op("Id",s2) - J*(op("adag",s1)*op("a",s2) + op("adag",s2)*op("a",s1))

        heff = - 0.5*im * gammas[j]*op("n",s1)*op("Id",s2)
        hj = h0 +  heff
        # troterized gate
        Gj = exp(-im * Δτ/2 * hj)
        push!(gates, Gj)
    end

    # The reverse gates for second order accuracy
    append!(gates, reverse(gates))

    return s, gates

end

function compute_trajectory_tebd(s::Vector{Index{Int64}}, gates::Vector{ITensor}, L::Int64, J::Float64,
                             Δτ::Float64, ttotal::Float64, Nh::Int64, gammas::Vector{Float64}, cutoff::Float64)
    # bettwe to put it in a function to avoid the global julia scope
    """ 
    s = indices for truncation
    Gj = TEBD circuit for one step
    L = chain size
    J = Hopping
    Δτ = timestep
    ttotal = final time
    cutoff = truncation cutoff
    Nh = local hilbert space dimension
    gamma = dissipation
    """
    # pre allocate memory
    n_tsteps = round(Int32, ttotal/Δτ)+1
    occupations = Vector{Vector{Float64}}(undef, n_tsteps)
    corr_matrices = Vector{Matrix{ComplexF64}}(undef, n_tsteps)
    entropies = zeros(n_tsteps)
    bond_dimensions = zeros(Int, n_tsteps)
    # for the wigner distributions
    a_operators = Vector{Vector{ComplexF64}}(undef, n_tsteps)

    # put a particle in the first site
    state_list = ["0" for n in 1:L]
    state_list[1] = "0"
    psi = MPS(s, state_list)
    psi_cand = copy(psi) # candidate for new state}
    norm = 0.0
    proba_act = 0.0
    dummy_counter = 1
    # throw all random nums at thestart for efficieny
    r1 = rand(length( 0.0:Δτ:ttotal))

    for t in 0.0:Δτ:ttotal
        # save observables of interest
        occupations[dummy_counter] = ITensorMPS.expect(psi,"N")    
        corr_matrices[dummy_counter] = correlation_matrix(psi,"adag","a")
        entropies[dummy_counter] = entangement_S(psi, round(Int64, L/2))
        bond_dimensions[dummy_counter] = ITensorMPS.maxlinkdim(psi)
        a_operators[dummy_counter] = ITensorMPS.expect(psi,"a") 

        # t≈ttotal && break

        # metropolis step
        psi_cand = apply(gates, psi; cutoff) # candidate for new state}
        norm = inner(psi_cand',psi_cand)
        proba_act = real(1 - norm)

        if r1[dummy_counter] > proba_act
            # here no jump so we accept the state
            psi = psi_cand/norm
        else
            # jump, so choose which state we project to
            psi = apply_jump_operator(s, gammas, psi, L)
        end

        dummy_counter+=1
    end

    return occupations, entropies, bond_dimensions, corr_matrices, a_operators
end


function apply_jump_operator(s_indices, dissipations, psi_mps, chain_l)
    # jump, so choose which state we project to
        δp_list = [dissipations[i]*ITensorMPS.expect(psi_mps,"N",sites=i) for i in 1:chain_l]  
        normalize!(δp_list)
        jump_site = ask_where(δp_list)
        jump_op = sqrt(dissipations[jump_site])*op("a",s_indices[jump_site]);
        psi_mps = apply(jump_op , psi_mps)
        
        return normalize!(psi_mps)

end

# ---------------
# MAIN  
# ---------------

println("Number of threads: ")
println(Threads.nthreads())

for F in F_list
    occupation_ensemble = Vector{Matrix{Float64}}(undef, N_trajectories)
    correlation_ensemble = Vector{Vector{Matrix{ComplexF64}}}(undef, N_trajectories)
    entropy_ensemble = zeros(N_trajectories, length(time_list))
    b_dims_ensemble = zeros(N_trajectories, length(time_list))
    a_ensemble = Vector{Vector{Vector{ComplexF64}}}(undef, N_trajectories)
    s_indices, gates = build_TEBD(L, ω, U, J, F, Delta_t, N, γ_list)
    timings = zeros(N_trajectories)

    println("Doing TEBD...")

    # use @elapsed to save the result of the time measurement macro

    Threads.@threads for traj in 1:N_trajectories
        timings[traj] = @elapsed occupations, Svns, bond_dim, correlation_matrix, a_operators = compute_trajectory_tebd(s_indices, gates, L, J,
                                Delta_t, maxt, N, γ_list, cutoff);
        occupations = reduce(hcat, occupations);
        occupation_ensemble[traj] = occupations
        entropy_ensemble[traj, 1:end] = Svns
        b_dims_ensemble[traj, 1:end] = bond_dim
        correlation_ensemble[traj] = correlation_matrix
        a_ensemble[traj] = a_operators
        
    end
    # 
    # average over trajectories when necessary
    mean_traj = mean(occupation_ensemble)
    mean_entropy = mean(entropy_ensemble,dims=1)[1,1:end]
    mean_bond = mean(b_dims_ensemble,dims=1)[1,1:end]
    tot_oc = sum(mean_traj, dims=1)[1:end]
    mean_corr = mean(correlation_ensemble)

    # vuild file and metadata
    str_file_name = @sprintf("../data/sims/benchmark/%s_N%i_L%i_om%.2f_J%.2f_gamma%.2f_kerr%.2f_drive%.2f_maxt%.2f_deltat%.2f_traj%i_cutexp%i.h5", 
                            evol_type,N,L,ω,J,γ,U,F,maxt,Delta_t, N_trajectories, cutoff_exponent)

    param_dict = Dict("type"=>evol_type,"N"=>N, "L"=>L, "omega" =>ω,"J" =>J,"gamma"=>γ ,"U"=>U ,"F"=>F ,
                    "maxt"=>maxt, "N_trajectories"=>N_trajectories, "cutoff"=>cutoff, "delta_t" =>Delta_t,
                    "time_per_trajctory" =>timings)

    # Save data ------------------------------------

    fid = h5open(str_file_name, "w")
        # write the metadata to its own group
        create_group(fid, "metadata")
        meta = fid["metadata"]
        for (k, v) in param_dict
                meta[k] = v  # save each entry under its key
        end
        # save the results
        res_g = create_group(fid, "results")
        res_g["occupations"] = mean_traj
        res_g["entropy_first_to_half"] = mean_entropy
        res_g["bond_dimension"] = mean_bond 
        # for the matrices we have to save them as a tensor where time is one of the indices
        res_g["twobody_correlation"] = cat(mean_corr...; dims=3)   

        size1 = length(a_ensemble)
        size2 = length(a_ensemble[1])
        size3 = length(a_ensemble[1][1])

        tn_ = Array{ComplexF64}(undef, size1, size2, size3);
        for i in 1:size1, j in 1:size2, k in 1:size3
            tn_[i, j, k] = a_ensemble[i][j][k]
        end

        res_g["annihilation_expectation"] = tn_     

    close(fid)
end

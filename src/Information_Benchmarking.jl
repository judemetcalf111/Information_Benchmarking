module Information_Benchmarking

using DrWatson
@quickactive "Information_Benchmarking"

using CUDA
using StatsBase
using Statistics

function calc_cumulative_info(H0_tot::Float32, H_k::CuArray{Float32, 2})
    # Sum the entropy across the entire GPU grid
    H_k_tot = sum(H_k)
    return H0_tot - H_k_tot
end

function calc_info_gain(I_k::Float32, I_k_minus_1::Float32)
    return I_k - I_k_minus_1
end

function calc_kldiv(G_k::CuArray{Float32, 2}, S_k::CuArray{Float32, 2}, eps::Float32=1e-7f0)
    # Ensure maps are strictly positive to avoid domain errors
    G_safe = G_k .+ eps
    S_safe = S_k .+ eps
    
    # Compute the element-wise divergence
    kl_map = @. G_safe * log(G_safe ) - log(S_safe)
    
    # Sum across the GPU array
    return sum(kl_map)
end

function calc_sim(G_k::CuArray{Float32, 2}, S_k::CuArray{Float32, 2})
    # Broadcast min function and sum the intersection
    return sum(min.(G_k, S_k))
end

function extract_nss_step(G_mu::Float64, G_sigma::Float64, fix_y::Int, fix_x::Int)
    # Extract the saliency value at the trajectory's current coordinate
    val = CUDA.@allowscalar G_static[fix_y, fix_x]
    
    return (val - G_mu) / G_sigma           # z-value of saliency
end

function calc_auc(G_k::CuArray{Float32, 2}, S_k::CuArray{Float32, 2})
    # Transfer GPU arrays to CPU for efficient sorting
    G_cpu = Array(G_k)
    S_cpu = Array(S_k)
    
    # Identify positive and negative locations based on the expanding density map
    pos_mask = G_cpu .> 0
    pos_vals = S_cpu[pos_mask]
    neg_vals = S_cpu[.!pos_mask]
    
    n_pos = length(pos_vals)
    n_neg = length(neg_vals)
    
    # Edge case: If there are no positive or negative samples, AUC is theoretically 0.5
    if n_pos == 0 || n_neg == 0
        return 0.5f0
    end
    
    # Combine values and compute ranks to derive the U statistic
    combined = vcat(pos_vals, neg_vals)
    ranks = invperm(sortperm(combined))
    
    # Calculate sum of ranks for the positive class
    sum_ranks_pos = sum(ranks[1:n_pos])
    
    # Calculate the Mann-Whitney U statistic and normalize to get AUC
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1) / 2)
    return u_stat / (n_pos * n_neg)
end


function calc_ess(trajectory_values::Vector{Float64})
    K = length(trajectory_values)
    
    # Compute autocorrelation for all possible lags
    # (In practice, you might restrict the maximum lag to K/2)
    lags = 1:(K-1)
    rhos = autocor(trajectory_values, lags)
    
    # Truncate the sum when autocorrelation drops below zero to reduce noise
    sum_rhos = 0.0
    for r in rhos
        if r < 0.0
            break
        end
        sum_rhos += r
    end
    
    ess = K / (1.0 + 2.0 * sum_rhos)
    return ess
end


###
### Idea, KL divergence on y-axis, and the Information on the x-axis. Showing the similarity gained from more information (lower entropy)
###

end

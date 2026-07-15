using Pkg
Pkg.activate(".")

using CUDA
using HDF5
using DataFrames
using StatsBase
using DrWatson

# ---------------------------------------------------------
# 1. Metric Definitions (Pure Julia, GPU-compatible)
# ---------------------------------------------------------
function calc_kldiv(G_k::CuArray{Float32, 2}, S_k::CuArray{Float32, 2}, eps::Float32=1e-7f0)
    G_safe = G_k .+ eps
    S_safe = S_k .+ eps
    return sum(G_safe .* log.(G_safe ./ S_safe))
end

function calc_sim(G_k::CuArray{Float32, 2}, S_k::CuArray{Float32, 2})
    return sum(min.(G_k, S_k))
end

function extract_nss_step(G_static::CuArray{Float32, 2}, fix_y::Int, fix_x::Int, mu::Float32, sigma::Float32)
    val = CUDA.@allowscalar G_static[fix_y, fix_x]
    return (val - mu) / sigma
end

function calc_entropy(S_k::CuArray{Float32, 2}, eps::Float32=1e-7f0)
    S_safe = S_k .+ eps
    return -sum(S_safe .* log.(S_safe))
end

# ---------------------------------------------------------
# 2. Main Active Accumulation Loop
# ---------------------------------------------------------
function evaluate_scanpath(filepath::String, grid_width::Int, grid_height::Int)
    
    # Load trajectory and target map from HDF5
    h5_data = h5open(filepath, "r")
    scanpath_x = read(h5_data, "scanpath_x")
    scanpath_y = read(h5_data, "scanpath_y")
    G_cpu = read(h5_data, "static_ground_truth") # The target landscape
    close(h5_data)
    
    # Pre-allocate static GPU arrays
    G_gpu = CuArray{Float32}(G_cpu)
    mu_G = mean(G_gpu)
    sigma_G = std(G_gpu)
    
    # Generate X, Y coordinate grids on GPU
    x_grid = CuArray{Float32}(repeat(1:grid_width, 1, grid_height)')
    y_grid = CuArray{Float32}(repeat(1:grid_height, grid_width, 1))
    
    # Initialize the dynamic knowledge map (Uniform Prior)
    N = grid_width * grid_height
    S_k = CUDA.fill(1.0f0 / N, (grid_height, grid_width))
    H_0 = calc_entropy(S_k)
    
    results = DataFrame(Step=Int[], KLD=Float32[], SIM=Float32[], NSS=Float32[], Info=Float32[], InfoGain=Float32[])
    
    I_k_minus_1 = 0.0f0
    
    for k in 1:length(scanpath_x)
        fix_x = scanpath_x[k]
        fix_y = scanpath_y[k]
        
        # 1. Compute Eccentricity and Foveal Pooling Radius
        E_gpu = sqrt.((x_grid .- fix_x).^2 .+ (y_grid .- fix_y).^2)
        r_gpu = 0.1f0 .* (E_gpu .+ 0.8f0)
        
        # 2. Dynamic Update (Bayesian map blending)
        # Here, a Gaussian masking function defines visual acuity based on r(E)
        mask = exp.(- (E_gpu.^2) ./ (2.0f0 .* r_gpu.^2))
        
        # Update knowledge: blend prior with ground truth G based on foveal mask
        # (Normalization step omitted for brevity; ensure S_k sums to 1.0)
        S_k .= (1.0f0 .- mask) .* S_k .+ mask .* G_gpu
        S_k ./= sum(S_k) 
        
        # 3. Calculate Metrics
        kld = calc_kldiv(G_gpu, S_k)
        sim = calc_sim(G_gpu, S_k)
        nss = extract_nss_step(G_gpu, fix_y, fix_x, mu_G, sigma_G)
        
        H_k = calc_entropy(S_k)
        I_k = H_0 - H_k
        info_gain = I_k - I_k_minus_1
        
        push!(results, (k, kld, sim, nss, I_k, info_gain))
        I_k_minus_1 = I_k
    end
    
    return results
end

# ---------------------------------------------------------
# 3. Execution and Export
# ---------------------------------------------------------
data_file = datadir("sims", "trajectory_data.h5")
df_results = evaluate_scanpath(data_file, 1920, 1080)

# Validate normality of heavy-tailed information gains 
# visually via qqnorm() before computing confidence intervals
using StatsPlots
qqnorm(df_results.InfoGain, title="Info Gain Q-Q", markercolor=:black)

# Save results to HDF5 using DrWatson 
safesave(datadir("results", "metrics_output.h5"), @strdict(df_results))

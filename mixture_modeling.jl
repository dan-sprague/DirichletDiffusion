using Turing, Distributions, LinearAlgebra, Random, Statistics
using CairoMakie
using StatsFuns: logsumexp
using FillArrays
using ADTypes
using Bijectors

# ==========================================
# 1. Global Settings & Data Generation
# ==========================================
Random.seed!(42)

# Experiment Params
const K_fit = 5           
const N_samples = 200      
const σ_true = 0.5         
const temperatures = [10.0, 1.0, 0.1, 0.01] 

println("--- Generating Data (σ = $σ_true) ---")

# True Means (3 Clusters)
μ_true = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
Σ_true = [1.0 0.0; 0.0 1.0] .* σ_true

# Generate Data
n_per_cluster = div(N_samples, 3)
X_data = hcat([rand(MvNormal(m, Σ_true), n_per_cluster) for m in μ_true]...)
X_data = X_data[:, shuffle(1:size(X_data, 2))]

println("Data Shape: $(size(X_data))")

# ==========================================
# 2. Vectorized Helpers
# ==========================================

function stick_breaking_transform(v::AbstractVector)
    K = length(v) + 1
    w = Vector{eltype(v)}(undef, K)
    remaining = one(eltype(v))
    for k in 1:(K-1)
        w[k] = v[k] * remaining
        remaining *= (1 - v[k])
    end
    w[K] = remaining
    return w
end

function sparsemax(z::AbstractVector; temperature=1.0)
    z_t = z ./ temperature
    D = length(z_t)
    z_sorted = sort(z_t, rev=true)
    cumsum_z = cumsum(z_sorted)
    k_vals = [1 + i * z_sorted[i] > cumsum_z[i] for i in 1:D]
    k = sum(k_vals)
    tau = (cumsum_z[k] - 1) / k
    return max.(0.0, z_t .- tau)
end

# ==========================================
# 3. Models (Unified Format)
# ==========================================

# --- 1. Standard GMM (Dirichlet) ---
@model function gmm_standard(x, K)
    D, N = size(x)

    # Prior for means (D-dimensional clusters)
    # Note: We use iteration to ensure we get a Vector of Vectors for MixtureModel
    μ ~ filldist(MvNormal(Zeros(D), I), K) 
    
    # Weights (Dirichlet)
    w ~ Dirichlet(K, 1.0)
    
    # Likelihood using MixtureModel
    # We split the matrix μ into a vector of vectors for the constructor
    distributions = [MvNormal(μ[:, k], I) for k in 1:K]
    x ~ MixtureModel(distributions, w)
end

# --- 2. Stick-Breaking GMM (Beta) ---
@model function gmm_stick_breaking(x, K)
    D, N = size(x)

    # Prior for means
    μ ~ filldist(MvNormal(Zeros(D), I), K) 
    
    # Stick-Breaking Process
    alpha ~ Gamma(1, 1.0)
    v ~ filldist(Beta(1, alpha), K - 1)
    w = stick_breaking_transform(v)
    
    # Likelihood using MixtureModel
    distributions = [MvNormal(μ[:, k], I) for k in 1:K]
    x ~ MixtureModel(distributions, w)
end

# --- 3. Sparsemax GMM ---
@model function gmm_sparsemax(x, K, temperature)
    D, N = size(x)

    # Prior for means
    μ ~ filldist(MvNormal(Zeros(D), I), K) 
    
    # Sparsemax Transformation
    ϕ ~ filldist(Normal(0, 1), K)
    w = sparsemax(ϕ; temperature=temperature)
    
    # Likelihood using MixtureModel
    # Add small epsilon to w to prevent numerical errors in MixtureModel logpdf if w is exactly 0
    w_safe = w .+ 1e-10
    w_norm = w_safe ./ sum(w_safe)
    
    distributions = [MvNormal(μ[:, k], I) for k in 1:K]
    x ~ MixtureModel(distributions, w_norm)
end

# ==========================================
# 4. Execution
# ==========================================

results_dict = Dict()

# Helper to extract mean vector of 'w' robustly from chains
function get_mean_weights(chain, sym)
    # Extracts the array of samples for the symbol (e.g., :w or :v) and computes column means
    vec(mean(Array(group(chain, sym)), dims=1))
end

# --- 1. Fit Standard ---
println("\nFitting Standard (Dirichlet)...")
chain_std = sample(gmm_standard(X_data, K_fit), NUTS(0.65), 500, progress=true)
w_std = get_mean_weights(chain_std, :w)
results_dict[:std] = w_std

# --- 2. Fit Stick-Breaking ---
println("\nFitting Stick-Breaking (Beta)...")
chain_sb = sample(gmm_stick_breaking(X_data, K_fit), NUTS(0.65), 500; adtype=AutoForwardDiff(), progress=true)
v_post = get_mean_weights(chain_sb, :v)
w_sb = stick_breaking_transform(v_post)
results_dict[:sb] = w_sb

# --- 3. Fit Sparsemax (Loop Temps) ---
results_dict[:sp] = []
for t in temperatures
    println("\nFitting Sparsemax (T=$t)...")
    chain_sp = sample(gmm_sparsemax(X_data, K_fit, t), NUTS(0.65), 500; adtype=AutoForwardDiff(), progress=true)
    ϕ_post = get_mean_weights(chain_sp, :ϕ)
    w_out = sparsemax(ϕ_post; temperature=t)
    push!(results_dict[:sp], w_out)
end

println("\nGenerating Plot...")

# ==========================================
# 4. Visualization (Discrete Matrix Construction)
# ==========================================
println("\nGenerating Discrete Matrix Heatmap...")

# 1. Setup Data
model_labels = [
    "Ground Truth",
    "Standard (Dirichlet)",
    "Stick-Breaking",
    "Sparsemax (T=$(temperatures[1]))", # T=1.0
    "Sparsemax (T=$(temperatures[2]))", # T=1.0
    "Sparsemax (T=$(temperatures[3]))"  # T=0.1
]

weights_raw = [
    [1/3,1/3,1/3,0.0,0.0],
    results_dict[:std],
    results_dict[:sb],
    results_dict[:sp][1],
    results_dict[:sp][2],
    results_dict[:sp][3]
]

N_models = length(weights_raw)

# 2. Construct the Discrete Matrix
# We define a "resolution" (e.g., 500 blocks wide) to represent the 0.0-1.0 probability space
resolution = 500 

# This matrix will hold the Rank ID (1, 2, 3...) for every "block" in the stick
# Dimensions: (Resolution x N_models) because heatmap expects (x, y)
discrete_data = zeros(Int, resolution, N_models)

for (model_idx, w_vec) in enumerate(weights_raw)
    # Sort weights descending (Rank 1 = Largest)
    w_sorted = sort(w_vec, rev=true)
    
    current_x = 1
    
    for (rank, weight) in enumerate(w_sorted)
        # Calculate how many "blocks" this cluster occupies
        n_blocks = round(Int, weight * resolution)
        
        # Determine end index, clamping to matrix bounds
        end_x = min(current_x + n_blocks - 1, resolution)
        
        # Fill the blocks with the Rank ID
        if end_x >= current_x
            discrete_data[current_x:end_x, model_idx] .= rank
        end
        
        current_x = end_x + 1
    end
    
    # Fill any tiny remaining gap due to rounding with the last active rank or 0
    if current_x <= resolution
        discrete_data[current_x:end, model_idx] .= length(w_sorted)
    end
end

# 3. Define Categorical Colors (Rank 1 to K_fit)
# We use a distinct palette so Rank 1 is clearly different from Rank 2
# Colors: Blue, Orange, Green, Purple, Red (example)
rank_colors = [:cornflowerblue, :orange, :mediumseagreen, :purple, :firebrick]

# 4. Plot
f = Figure(size = (1000, 300), backgroundcolor = :white)

ax = Axis(f[1, 1],
    title = "Cluster Weight Dominance (Discrete Proportions)",
    yticks = (1:N_models, model_labels),
    yreversed = true, # Top model is index 1
    aspect = 5.0,     # Requested 5:1 Aspect Ratio
    xgridvisible = false, ygridvisible = false,
    xticksvisible = false, yticksvisible = false
)

# We use heatmap, which maps the integers in `discrete_data` to `colormap`
hm = heatmap!(ax, 1:resolution, 1:N_models, discrete_data,
    colormap = rank_colors,
    colorrange = (1, K_fit) # Ensure integers 1..K map to the specific colors
)

# 5. Legend
# Manually create legend entries matching the rank_colors
leg_elements = [PolyElement(color = rank_colors[i], strokecolor = :transparent) for i in 1:K_fit]
leg_labels = ["Group $i" for i in 1:K_fit]
Legend(f[1, 2], leg_elements, leg_labels, "Cluster ID", framevisible=false)

save("gmm_discrete_heatmap.png", f)
println("Done. Plot saved to gmm_discrete_heatmap.png")
display(f)